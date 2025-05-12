import pandas as pd
import ollama
import os
import re
from typing import Dict, Any, Optional, List, Tuple
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, OptimizersConfigDiff, Filter, FieldCondition, MatchValue, Range
import logging
import random
from datetime import datetime
from dotenv import load_dotenv
import json
import hashlib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PowerOutageAssistant:
    def __init__(self, data_folder: str):
        self.data_folder = data_folder
        self.customer_df = self.load_csv('CIS.csv')
        self.oms_df = self.load_csv('OMS.csv')
        self.ami_df = self.load_csv('AMI.csv')
        self.scada_df = self.load_csv('SCADA.csv')
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
        self.texts_file = os.path.join(data_folder, f"texts_{os.getenv('EMBEDDING_MODEL')}.pkl")
        
        if all(df is not None for df in [self.oms_df, self.ami_df, self.scada_df]):
            try:
                model_name = os.getenv('EMBEDDING_MODEL')
                token = os.getenv('HF_TOKEN', None)
                self.model = SentenceTransformer(model_name, use_auth_token=token)
                self.qdrant_client = QdrantClient(os.getenv('QDRANT_URL', "http://localhost:6333"))
                self.texts = self.initialize_vector_store()
            except Exception as e:
                logging.error(f"RAG init failed for model {model_name}: {str(e)[:200]}")
                self.texts, self.qdrant_client = [], None
        else:
            self.texts, self.qdrant_client = [], None

    def load_csv(self, filename: str) -> Optional[pd.DataFrame]:
        """Load CSV"""
        try:
            df = pd.read_csv(os.path.join(self.data_folder, filename))
            date_parsers = {
                'CIS.csv': [('ValidFrom', '%d-%m-%Y'), ('ValidTo', '%d-%m-%Y')],
                'OMS.csv': [('Date', '%d-%m-%Y')],
                'AMI.csv': [('ReadingTimestamp', '%d-%m-%Y %H:%M')],
                'SCADA.csv': [('DetectedTime', '%d-%m-%Y %H:%M')]
            }
            for col, fmt in date_parsers.get(filename, []):
                df[col] = pd.to_datetime(df[col], format=fmt, errors='coerce')
            return df
        except Exception as e:
            logging.error(f"Failed to load {filename}: {str(e)[:100]}...")
            return None
        
    def initialize_vector_store(self) -> List[str]:
        """Optimized vector store init with hybrid search and fault tolerance"""
        from config.config import qdrant_collection_name

        # self.qdrant_client.delete_collection(qdrant_collection_name)

        texts = self._create_text_representations(qdrant_collection_name)

        if not texts: 
            # logging.info("No new texts to index")
            return []
        
        # print(texts)
        try:
            self._update_or_create_embeddings(qdrant_collection_name, texts)
            return [text for text, _, _, _ in texts]
        except Exception as e:
            logging.error(f"Vector store initialization failed: {e}")
            return []

    def _create_text_representations(self, collection_name: str) -> List[Tuple[str, int, str, str]]:
        from config.config import templates

        try:
            existing_hashes = self._get_existing_hashes(collection_name)
            texts = []
            record_id_counter = 1

            for df, prefix in [
                (self.oms_df, 'OMS'), 
                (self.ami_df, 'AMI'), 
                (self.scada_df, 'SCADA')
            ]:
                if df is not None:
                    df = df.copy()
                    if prefix == 'OMS':
                        df = df.rename(columns={"Duration (hrs)": "Duration_hrs"})
                    if prefix == 'OMS' and 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce').dt.strftime('%Y-%m-%d')
                    if prefix == 'AMI' and 'ReadingTimestamp' in df.columns:
                        df['ReadingTimestamp'] = pd.to_datetime(df['ReadingTimestamp'], format='%d-%m-%Y %H:%M', errors='coerce').dt.strftime('%Y-%m-%d')
                    if prefix == 'SCADA' and 'DetectedTime' in df.columns:
                        df['DetectedTime'] = pd.to_datetime(df['DetectedTime'], format='%d-%m-%Y %H:%M', errors='coerce').dt.strftime('%Y-%m-%d')

                    df = df.fillna('unknown')

                    def compute_row_hash(row):
                        row_dict = row.dropna().to_dict()
                        row_json = json.dumps(row_dict, sort_keys=True, default=str)
                        return hashlib.sha256(row_json.encode('utf-8')).hexdigest()

                    df['row_hash'] = df.apply(compute_row_hash, axis=1)
                    df['record_id'] = range(record_id_counter, record_id_counter + len(df))  
                    record_id_counter += len(df)

                    for _, row in df.iterrows():
                        record_id = row['record_id']
                        row_hash = row['row_hash']
                        if str(record_id) in existing_hashes and existing_hashes[str(record_id)] == row_hash:
                            continue

                        try:
                            text = templates[prefix].format(**row.to_dict())
                            texts.append((text, record_id, prefix, row_hash))
                        except KeyError as e:
                            logging.warning(f"Template formatting failed for {record_id}: {e}")
                            continue
            
            if texts: logging.info(f"Generated {len(texts)} new text representations")
            return texts
        except Exception as e:
            logging.error(f"Text representation creation failed: {e}")
            return []
    
    def _get_existing_hashes(self, collection_name: str) -> Dict[str, str]:
        """Fetch existing point hashes from Qdrant with pagination."""

        try:
            if not self.qdrant_client.collection_exists(collection_name):
                return {}
            
            points = []
            offset = None
            while True:
                batch, next_offset = self.qdrant_client.scroll(
                    collection_name=collection_name,
                    scroll_filter=None,
                    limit=1000,
                    with_payload=True,
                    with_vectors=False,
                    offset=offset
                )
                points.extend(batch)
                if not next_offset:
                    break
                offset = next_offset

            hashes = {str(p.id): p.payload.get("record_hash", "") for p in points}
            # logging.info(f"Fetched {len(hashes)} existing hashes from {collection_name}")
            return hashes

        except Exception as e:
            logging.error(f"Error fetching existing hashes: {e}")
            return {}

    def _update_or_create_embeddings(self, collection_name: str, texts: List[Tuple[str, int, str, str]]) -> None:
        """Update or create Qdrant collection with batched embeddings."""

        try:
            if not texts:
                return
            
            batch_size = 1000
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                embeddings = self.model.encode(
                    [text for text, _, _, _ in batch_texts],
                    batch_size=64,
                    show_progress_bar=False
                )
                
                points = []
                for (text, record_id, source, hash_code), embedding in zip(batch_texts, embeddings):
                    date_match = re.search(r'(on|Detected|Timestamp)\s*:\s*(\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
                    customer_id_match = re.search(
                        r'(?:Affected Customers IDs: \[([^\]]*)\]|Customer ID (\d+)|ID (\d+))',
                        text,
                        re.IGNORECASE
                    )

                    customer_ids = []
                    if customer_id_match:
                        if customer_id_match.group(1):  
                            customer_ids = [id.strip() for id in customer_id_match.group(1).split(",") if id.strip().isdigit()]
                        elif customer_id_match.group(2): 
                            customer_ids = [customer_id_match.group(2)]

                    date_timestamp = 0
                    if date_match:
                        try:
                            date_str = date_match.group(2)
                            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                            date_timestamp = int(date_obj.timestamp())
                        except ValueError as e:
                            logging.warning(f"Failed to parse date {date_str}: {e}")

                    points.append(PointStruct(
                        id=record_id,
                        vector=embedding.tolist(),
                        payload = {
                                "text": text,
                                "source": source,
                                "date": date_timestamp,
                                "customer_ids": customer_ids,
                                "record_hash": hash_code,
                            }
                    ))
        
                if not self.qdrant_client.collection_exists(collection_name):
                    self.qdrant_client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=embeddings.shape[1],
                            distance=Distance.COSINE,
                            on_disk=True
                        ),
                        optimizers_config=OptimizersConfigDiff(
                            indexing_threshold=10_000
                        )
                    )
                    logging.info(f"Created new Qdrant collection: {collection_name}")

                self.qdrant_client.upsert(collection_name=collection_name, points=points)
                logging.debug(f"Upserted {len(points)} points in batch")

            logging.info(f"Updated {collection_name} with {len(texts)} embeddings")
        except Exception as e:
            logging.error(f"Embedding update failed: {e}")
            raise
        
    def retrieve_documents(self, query: str, context: Dict[str, Any], temporal_dates: List[str], k: int = 5) -> List[Dict]:
        """Retrieve text chunks based on query and context with exact date filtering."""
        from config.config import qdrant_collection_name

        if not self.qdrant_client:
            logging.warning("Vector store offline")
            return []

        try:
            user_id = str(context.get('id', 'unknown'))
            context_text = self.format_context(context)
            combined_input = f"{query}\n\n{context_text}"
            query_embedding = self.model.encode([combined_input], show_progress_bar=False)[0]

            filters = []
            date_filters = []
            if temporal_dates:
                try:
                    for query_date in temporal_dates:
                        date_obj = datetime.strptime(query_date, '%Y-%m-%d')
                        date_timestamp = int(date_obj.timestamp())
                        date_filters.append(
                            FieldCondition(
                                key="date",
                                match=MatchValue(
                                    value=date_timestamp 
                                )
                            )
                        )
                except ValueError as e:
                    logging.warning(f"Failed to parse query_date {query_date}: {e}")
            
            if user_id != 'unknown':
                filters.append(
                    FieldCondition(
                        key="customer_ids",
                        match=MatchValue(value=user_id)
                    )
                )

            filter_condition = Filter(
                must=filters, 
                should=date_filters if date_filters else None  
            ) if filters or date_filters else None

            results = self.qdrant_client.query_points(
                collection_name=qdrant_collection_name,
                query=query_embedding.tolist(),
                limit=k,
                with_payload=True,
                query_filter=filter_condition
            )

            filtered_results = []
            for doc in results.points:
                text = doc.payload.get('text', '')
                logging.debug(f"Initial chunk: {text[:100]} (score: {doc.score})")
                if not text or re.search(r'\d{3}-\d{4}', text):
                    continue
                filtered_results.append({
                    "text": text,
                    "score": doc.score,
                    "metadata": {
                        "source": doc.payload.get('source', 'Unknown'),
                        "customer_match": user_id in text,
                        "date_match": any(query_date in text for query_date in temporal_dates)
                    }
                })
                logging.debug(f"Retrieved chunk: {text[:100]} (score: {doc.score})")

            return filtered_results[:k]

        except Exception as e:
            logging.error(f"Retrieval error: {str(e)[:200]}")
            return []

    def format_context(self, context: Dict[str, Any]) -> str:
        """Convert structured customer context into concise, LLM-friendly format."""
        return f"""
                Customer ID: {context.get('id', 'N/A')}
                Customer Name: {context.get('customer_name', 'N/A')}
                Status: {context.get('disconnection_reason', 'N/A')}
                Account Active: {context.get('active', False)}
                Service Address: {context.get('service_address', 'N/A')}
                Location: ({context.get('latitude', 0.0)}, {context.get('longitude', 0.0)})
                Contact: {context['contact_info'].get('email', 'N/A')} | {context['contact_info'].get('phone', 'N/A')}
                Account Validity: {context['account_dates'].get('valid_from', 'N/A')} to {context['account_dates'].get('valid_to', 'N/A')}
                Query Date: {context.get('current_timestamp', 'N/A')}
            """

    def fetch_customer_context(self, user_id: str) -> Dict[str, Any]:
        """Get complete customer info without nearby outages"""
        if self.customer_df is None:
            return {"error": "Customer system unavailable", "active": False}
        
        try:
            customer_rows = self.customer_df[self.customer_df['CustomerID'] == int(user_id)]
            if customer_rows.empty:
                return {"error": f"No customer found for ID {user_id}", "active": False}
            
            customer = customer_rows.iloc[0]
            context = {
                'id': customer.get('CustomerID', ''),
                'active': customer.get('AccountStatus', '') == 'Active' and customer.get('IsEnabled', False),
                'disconnection_reason': customer.get('AccountStatus', 'unknown'),
                'service_address': f"{customer.get('Address', '')}, {customer.get('City', '')}, {customer.get('State', '')} {customer.get('ZipCode', '')}",
                'latitude': float(customer.get('Latitude', 0)),
                'longitude': float(customer.get('Longitude', 0)),
                'customer_name': customer.get('CustomerName', ''),
                'contact_info': {
                    'email': customer.get('Email', ''),
                    'phone': customer.get('MobileNumber', '')
                },
                'account_dates': {
                    'valid_from': customer.get('ValidFrom', '').strftime('%Y-%m-%d') if pd.notnull(customer.get('ValidFrom')) else '',
                    'valid_to': customer.get('ValidTo', '').strftime('%Y-%m-%d') if pd.notnull(customer.get('ValidTo')) else ''
                },
                'current_timestamp': datetime.now().strftime('%Y-%m-%d')
            }
            return context
        except Exception as e:
            logging.error(f"Customer lookup failed for ID {user_id}: {str(e)[:200]}")
            return {"error": f"Customer lookup failed: {str(e)[:200]}", "active": False}

    def detect_intent_with_llama(self, user_input: str) -> str:
        """Detect the intent of the user's input using LLaMA."""
        from config.config import user_intent_prompt

        try:
            response = ollama.chat(
                model="llama3.2",
                messages=[{
                    "role": "system", 
                     "content": "You classify user intent based on message."},
                    {"role": "user", 
                     "content": user_intent_prompt.format(
                                    user_input=user_input
                                )}
                ],
                options={"temperature": 0.4}
            )
            return response["message"]["content"].strip().lower()
        except Exception as e:
            logging.error(f"Intent detection failed: {e}")
            return "other"

    def ask_feedback(self) -> str:
        """Ask for user feedback."""
        return "Did we resolve your query? Feel free to let me know."

    def generate_llama_response(self, context: Dict[str, Any], user_input: str) -> str:
        """Generate LLaMA response by passing context and user input to the model."""
        import logging
        import random
        import ollama
        import re
        import ast
        from config.config import outage_assistant_prompt, temporal_query_prompt

        ticket_number = f"SR-{random.randint(10000, 99999)}"
        # logging.info(f"Generated ticket number: {ticket_number}")

        if not context.get('active', False):
            return f"{context.get('customer_name', 'Customer')}, your account is not active. Would you like to raise a service request ({ticket_number})?"

        try:
            temporal_query_prompt = temporal_query_prompt.format(
                                    current_timestamp=context.get('current_timestamp'),
                                    user_input=user_input
                                )
            
            # logging.info("Temporal Query Prompt:\n")
            # print(f"'''\n{temporal_query_prompt}\n'''\n")

            response = ollama.chat(
                model="llama3.2",
                messages=[
                    {"role": "user", 
                     "content": temporal_query_prompt}
                ],
                options={"temperature": 0.1}
            )

            response = response['message']['content']
            temporal_dates = []
            if response and response.strip() != "[]":
                try:
                    parsed_content = ast.literal_eval(response)
                    # print(parsed_content)
                    if parsed_content:
                        temporal_dates = [item["interpreted"] for item in parsed_content]
                        context['temporal_intent'] = parsed_content[0].get('intent')
                except (ValueError, SyntaxError) as e:
                    logging.error("Failed to parse model output:", e)

            retrieved_docs = self.retrieve_documents(user_input, context, temporal_dates, k=50)
            retrieved_context = "\n".join(doc['text'] for doc in retrieved_docs) if retrieved_docs else "No relevant data found."
            retrieved_context = re.sub(r'\(?\d+\.\d+,\s?-?\d+\.\d+\)?', '', retrieved_context)

            llama_prompt = outage_assistant_prompt.format(
                        customer_id=context.get('id', 'unknown'),
                        customer_name=context.get('customer_name', 'Customer'),
                        account_status=context.get('active', 'unknown'),
                        service_address=context.get('service_address', 'unknown'),
                        Latitude=context.get('latitude', '0.0'),
                        Longitude=context.get('longitude', '0.0'),
                        current_timestamp=context.get('current_timestamp'),
                        temporal_intent=context.get('temporal_intent', 'present'),
                        retrieved_context=retrieved_context,
                        temporal_context=parsed_content,
                        ticket_number=ticket_number,
                        user_input=user_input
                    )
            
            # logging.info("Llama prompt:\n")
            # print(f"'''\n{llama_prompt}\n'''\n")
            # return

            # Generate response using LLaMA model, passing the context, retrieved documents, and user input
            response = ollama.chat(
                model="llama3.2",
                messages=[{
                    "role": "user",
                    "content": llama_prompt
                }],
                options={
                    "temperature": 0.3,
                    "max_tokens": 100,
                }
            )

            # Clean up and return the LLaMA-generated response
            clean_response = response['message']['content'].strip()
            if not clean_response:
                clean_response = f"{context.get('customer_name', 'Customer')}, no outages or faults found today. Would you like to raise a service request ({ticket_number})?"
            return clean_response

        except Exception as e:
            logging.error(f"Response generation failed: {str(e)[:200]}")
            return f"{context.get('customer_name', 'Customer')}, system error. Would you like to raise a service request ({ticket_number})?"

    def chat(self, user_input: str, user_id: str):
        """Handle user chat and guide them based on intent."""
        try:
            user_id = int(user_id)

            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = {
                    "intent": None,
                    "pending_action": None,
                    "data": {}
                }
            session = self.user_sessions[user_id]

            context = self.fetch_customer_context(user_id)
            if "error" in context:
                ticket_number = f"SR-{random.randint(10000, 99999)}"
                return {
                    "status": "error",
                    "message": f"Customer not found. Would you like to raise a service request ({ticket_number})? Did we resolve your query? Feel free to let me know."
                }

            if session["intent"] is None:
                intent = self.detect_intent_with_llama(user_input)
                session["intent"] = intent
            else:
                intent = session["intent"]

            if intent == "outage_report":
                response = self.generate_llama_response(context, user_input)
                return {"status": "success", "message": response}

            if intent == "change_email":
                if session["pending_action"] is None:
                    session["pending_action"] = "ask_new_email"
                    return {
                        "status": "need_input",
                        "message": "Please enter your new email address."
                    }
                elif session["pending_action"] == "ask_new_email":
                    new_email = user_input.strip()
                    if self.is_valid_email(new_email):
                        old_email = self.customer_df.loc[self.customer_df['CustomerID'] == user_id, 'Email'].values[0]
                        if new_email == old_email:
                            self.user_sessions.pop(user_id, None)
                            return {"status": "success",
                                    "message": "The new email is the same as the current email. No change made."}
                        else:
                            self.customer_df.loc[self.customer_df['CustomerID'] == user_id, 'Email'] = new_email
                            self.customer_df.to_csv(os.path.join(self.data_folder, 'CIS.csv'), index=False)
                            self.user_sessions.pop(user_id, None)
                            return {"status": "success", 
                                    "message": f"Your email has been successfully updated to {new_email}. " + self.ask_feedback()}
                    else:
                        return {"status": "error", 
                                "message": "The email you entered is invalid. Please enter a valid email address."}

            return {"status": "pending", "message": "Please follow the necessary steps."}

        except Exception as e:
            logging.error(f"Chat processing failed: {str(e)[:200]}")
            ticket_number = f"SR-{random.randint(10000, 99999)}"
            return {
                "status": "error",
                "message": f"An error occurred. Would you like to raise a service request ({ticket_number})? Did we resolve your query? Feel free to let me know."
            }

if __name__ == "__main__":
    assistant = PowerOutageAssistant('data')

    queries = [
        # 'Why is there no power at my house today?',
        'Is there any planned outage tomorrow?',
        # 'Will there be an outage on May 12, 2025?',
        # 'How many outages were caused by storms this year?',
        # 'My house has been experiencing a power cut for the last 2 hours.',
        # 'I think there is a fault, the lights are not turning on.',
        # 'No response from smart meter and no electricity.',
        # 'Is there scheduled maintenance at my address tomorrow?',
        # 'Was there an outage on February 30, 2025?',
        # 'Why was there no power sometime last week?',
        # "Will there be an outage tomorrow or next Friday?"
    ]
    user_id = 1008

    temporal_test_queries = [
        "Why is there no power at my house today?",
        "Will there be an outage tomorrow or the day after?",
        "Please check if something is planned on the upcoming Friday and next Tuesday.",
        "Is there any power cut scheduled in 3 days or 2 weeks?",
        "What about maintenance this Sunday or last week?",
        "Has there been any interruption since yesterday or earlier this week?",
        "Can you confirm if any work is planned for today or next month?",
        "Check if the supply will be back by this weekend or the next one.",
        "Outage on June 5th or the following Monday?",
        "Is there any downtime planned between now and next Wednesday?",
        "Will I have electricity on the 25th or 3 days from now?"
    ]

    for query in temporal_test_queries:
        response = assistant.chat(query, user_id)
        print(f"\n[Customer {user_id}] Query: '{query}'\nResponse:", response["message"], '\n')
