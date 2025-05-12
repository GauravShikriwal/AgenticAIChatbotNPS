# config/prompts.py
qdrant_collection_name = "power_outage_vectors_v1"

templates = {
        'OMS': (
            "[OMS] Outage report (ID: {OutageID}) dated on: {Date}. The outage type is '{Type}' caused by '{Cause}', estimated to last {Duration_hrs} hours. " 
            "Current status: {Status}. Affected Customers IDs: {CustomersAffected}."
        ),
        'AMI': (
            "[AMI] Smart Meter {SmartMeterID} for Customer ID {CustomerID} reported status '{CommunicationStatus}' with voltage reading {Voltage} V. "
            "Last recorded timestamp: {ReadingTimestamp}."
        ),
        'SCADA': (
            "[SCADA] Fault detected on {AssetType} {AssetID} located at coordinates ({Latitude}, {Longitude}). "
            "Type: {FaultType}, Detected: {DetectedTime}, Resolved: {Resolved}. Affected Customers IDs: {CustomersAffected}."
        )
    }

user_intent_prompt = (
    "You are an intelligent assistant in the energy utility domain. A user has sent the following message:\n\n"
    "\"{user_input}\"\n\n"
    "Classify the user's intent into one of these categories:\n"
    "- billing_query\n"
    "- payment_issue\n"
    "- outage_report\n"
    "- service_request\n"
    "- change_email\n"
    "- change_password\n"
    "- change_address\n"
    "- change_mobile_number\n"
    "- other\n\n"
    "Respond only with the intent label"
)

temporal_query_prompt = (
    "You are a date interpretation engine.\n"
    "Identify and convert natural language time expressions in the input to calendar dates and determine if the query refers to a **past**, **present**, or **future** event..\n"
    "Use the provided 'Today Date' to interpret terms like 'today', 'tomorrow', 'next Friday', 'in 3 days', etc.\n\n"
    "Rules:\n"
    "- Return ONLY a JSON list: [{{original': 'phrase', 'interpreted': 'YYYY-MM-DD', 'intent': 'past' | 'present' | 'future'}}]\n"
    "- 'original' must contain ONLY the exact time expression (e.g., 'tomorrow', 'next Friday').\n"
    "- Do NOT return code, explanations, or any text outside the JSON list.\n"
    "- If no time expressions are found, return an empty list: []\n"
    "- Use the exact phrase from the input as 'original'.\n"
    "- Ensure 'interpreted' is in YYYY-MM-DD format.\n\n"
    "Today Date: {current_timestamp}\n"
    "Input: {user_input}\n"
    "Output:"
)

outage_assistant_prompt = (
    "You are a polite, empathetic virtual assistant for a power utility company.\n"
    "Your role is to answer all queries of user related to power outage or lack of electricity.\n\n"
    "Get all factual information strictly from the verified information provided below as Customer Details and Retrieved Context.\n"
    "Do not invent or assume any information. Use only what is explicitly mentioned.\n\n"

    "Customer Details:\n"
    "- ID: {customer_id}\n"
    "- Name: {customer_name}\n"
    "- Status: {account_status}\n"
    "- Address: {service_address}\n"
    "- Latitude: {Latitude}\n"
    "- Longitude: {Longitude}\n"
    "- Reference Date: {current_timestamp}\n"
    "- Temporal Query Intent: {temporal_intent}\n\n"

    "Retrieved Context:\n"
    "{retrieved_context}\n\n"

    "Temporal Context:\n"
    "{temporal_context}\n\n"

    "Response Requirements and Reasoning Flow:\n"
    "- Use a professional, confident, and human-like tone. Avoid robotic or vague phrasing.\n"
    "- Limit your response to a maximum of 80 words.\n"
    # "- If the Retricontext has no relevant data, explain that no planned outages were found for the requested days."
    "- Never reference or use information related to other customers, even if present in the context.\n"
    "- Important: Use the provided 'Reference Date' to interpret natural language time expressions like 'today', "
    "'tomorrow' or specific dates and never specifically mention 'Reference Date' term.\n"
    "- Strictly follow the decision flow below. As soon as a condition matches (i.e., answer is Yes), stop and respond accordingly. Do not proceed to further steps.\n\n"

    "Flow:\n"
    "1. If an outage affecting the customer's address is found in [OMS], explain the reason and give the estimated restoration time (ETA).\n"
    "2. If Smart Meter data in [AMI] shows 'No communication' or 'Voltage = 0', advise the customer to check their MCB or DB.\n"
    "3. If a [SCADA] issue affects the customer, explain the reason and give the estimated restoration time (ETA) if available.\n"
    # "4. If GIS data shows upstream fault or planned maintenance, explain that and raise a Service Request.\n"
    "5. If none of the above apply, politely inform the customer that you'll raise a Service Request (Ticket No: {ticket_number}) for further investigation.\n\n"

    "- Never describe the above steps in your response.\n"

    "Customer Query: {user_input}\n"
    "Respond:"
)

# docker run -d -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
