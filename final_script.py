import mysql.connector
import faiss
import numpy as np
import pickle
import requests
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import json

# Step 1: Database Connection
def connect_to_db():
    return mysql.connector.connect(
        host="3.6.18.112",
        user="test_user",
        password="testP@ssw0rd112!",
        database="ienergy_ehs_live"
    )

# Step 2: Extract Data from Database
def extract_data(connection, table_name, text_columns):
    cursor = connection.cursor(dictionary=True)
    cursor.execute(f"SELECT id, {', '.join(text_columns)} FROM {table_name}")
    data = cursor.fetchall()
    cursor.close()
    return data

# Step 3: Generate Embeddings
def generate_embeddings(data, text_columns, model):
    embeddings = []
    id_mapping = []
    for row in data:
        combined_text = " ".join([row[col] for col in text_columns if row[col]])
        embedding = model.encode(combined_text)
        embeddings.append(embedding)
        id_mapping.append(row["id"])
    return np.array(embeddings), id_mapping

# Step 4: Save Embeddings in FAISS
def save_to_faiss(embeddings, id_mapping, faiss_index_path, id_mapping_path):
    d = embeddings.shape[1]  # Dimension of embeddings
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    
    # Save FAISS index
    faiss.write_index(index, faiss_index_path)
    
    # Save ID mapping
    with open(id_mapping_path, "wb") as f:
        pickle.dump(id_mapping, f)

# Step 5: Query Decomposition Using LLM
def decompose_query(columns, user_query):
    # gemini_url = "https://gemini-api.example.com/v1/query"
    # headers = {"Authorization": "Bearer GEMINI_API_KEY", "Content-Type": "application/json"}

    genai.configure(api_key='AIzaSyCWZQdpgOkONxAM0kNCaeAmlqLXsH5yzOA')
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = (
        """Now what we have to do is I will provide you with a query. You will have to divide that query into what can be answered directly by generating a SQL query from the table and what can’t be .
        For example a table has four columns : job_type, insurance_type, summary_of_incident and time_of_day. Now imagine a query is “Filter out the incidents that happened after 4 pm with no insurance and construction job where the boy fell from the top”. Now we can filter out the rows by job_type= construction, insurance_type=no, time_of_day>1600 hours. But we can’t filter out using a SQL query for summary_of_incident=boy fell from top as it will be different for each incident/row.

        So divide the query into Query 1- [what can be separated using a sql query] and
        Query 2- [what needs similarity search using different methods]. Query 2 should just contain the part of the prompt left, no other text. SQL query will be generated from Query 1 and further methods will be used for Query 2 (You dont have to do this step).
        example output for query 2 , (prompt is same as above)- {the boy fell from the top}

        Now give the output - {Query 1},{Query 2} in json format for the following columns of table ehs_incident_details\n"""
        f'''Table columns - int(11)
NO
PRI
NULL
auto_increment
incidentId
varchar(255)
YES
type
varchar(255)
YES
incidentDate
datetime
YES
NULL
reportDate
datetime
YES
NULL
functionData
varchar(255)
YES
severity
varchar(255)
YES
siteId
int(11)
YES
MUL
NULL
roId
int(11)
YES
MUL
NULL
areaId
int(11)
YES
MUL
NULL
subAreaId
varchar(255)
YES
MUL
NULL
otherSubArea
varchar(150)
YES
NULL
activityId
int(11)
YES
MUL
NULL
subActivityId
varchar(255)
YES
MUL
NULL
investigationId
int(11)
YES
MUL
NULL
so_recommended_investigation
tinyint(4)
NO
0
staffType
varchar(50)
YES
NULL
staffEmployer
varchar(50)
YES
NULL
staffName
int(11)
YES
MUL
NULL
otherStaffName
varchar(150)
YES
NULL
severityPostReview
varchar(255)
YES
reportAtLabel
varchar(50)
YES
NULL
fireInvolved
tinyint(1)
YES
0
alcoholViolator
tinyint(1)
YES
0
ppeViolator
tinyint(1)
YES
0
trafficViolation
tinyint(4)
YES
0
unfitBuyerTruck
tinyint(1)
YES
0
unfitHEMM
tinyint(1)
YES
0
inundation
tinyint(4)
YES
0
fallOfObject
tinyint(4)
NO
0
facilityOutage
tinyint(1)
YES
0
permanentDisability
tinyint(1)
YES
0
facilityShutdown
tinyint(1)
YES
0
explosion
tinyint(1)
YES
0
longDurationFire
tinyint(1)
YES
0
equipmentFailure
tinyint(1)
YES
0
indirectLosses
varchar(255)
YES
facilityStatus
varchar(255)
YES
similarIncidentHappened
varchar(255)
YES
summary
varchar(255)
YES
description
text
YES
NULL
measuresOrActions
text
YES
NULL
fatalityEmployees
int(10) unsigned
YES
0
fatalityContractors
int(10) unsigned
YES
0
fatalityOthers
int(10) unsigned
YES
0
injuredEmployees
int(10) unsigned
YES
0
injuredContractors
int(10) unsigned
YES
0
injuredOthers
int(10) unsigned
YES
0
lostManhoursEmployees
int(10) unsigned
YES
0
lostManhoursContractors
int(10) unsigned
YES
0
lostManhoursOthers
int(10) unsigned
YES
0
reportedBy
int(11)
YES
MUL
NULL
reportedAt
datetime
YES
NULL
peopleCount
int(10) unsigned
YES
0
directIncidentLoss
int(10) unsigned
YES
0
directIncidentLossToEquipment
tinyint(1)
YES
0
insuranceClaimRaised
tinyint(1)
YES
0
insuranceApplicable
tinyint(1)
YES
NULL
insuranceClaimAmount
int(10) unsigned
YES
0
insuranceApprovedAmount
int(10) unsigned
YES
0
insuranceUnapprovedAmount
int(10) unsigned
YES
0
insuranceFileId
int(11)
YES
MUL
NULL
cause
text
YES
NULL
fireCause
text
YES
NULL
avoidable
varchar(255)
YES
No
couldAvoidBy
text
YES
NULL
spillCause
varchar(255)
YES
rootCause
varchar(255)
YES
environmentalImpact
varchar(255)
YES
No
environmentalAssessment
text
YES
NULL
mitigationMeasures
text
YES
NULL
solutionType
varchar(255)
YES
solutions
text
YES
NULL
rootCauseDescriptions
text
YES
NULL
typeOnReportTime
varchar(255)
YES
typeAfterInvestigation
varchar(255)
YES
investigationCompleted
tinyint(1)
YES
0
reviewCompleted
tinyint(1)
YES
0
closureCompleted
tinyint(1)
YES
0
investigationCompletionSummary
text
YES
NULL
reviewCompletionSummary
text
YES
NULL
closureCompletionSummary
text
YES
NULL
investigationCompletionDate
date
YES
NULL
investigationReviewedBy
int(11)
YES
NULL
investigationReviewedDate
datetime
YES
NULL
mmReviewedInvestigation
tinyint(4)
YES
NULL
investigationReviewedByMMDate
datetime
YES
NULL
investigationReviewedByMM
int(11)
YES
NULL
investigationReviewed
tinyint(4)
YES
NULL
reviewCompletionDate
date
YES
NULL
closureCompletionDate
date
YES
NULL
incidentStatus
varchar(255)
YES
draft
isPersonalInjury
tinyint(1)
YES
NULL
ltiHoursUpdate
tinyint(4)
NO
0
ltiHoursUpdatedAt
date
YES
NULL
ltiHoursUpdatedBy
int(11)
YES
NULL
ltiUpdatedAfterStatus
varchar(100)
YES
NULL
isVehicleIncident
tinyint(1)
YES
NULL
isSpillsIncident
tinyint(1)
YES
NULL
isAssetIncident
tinyint(1)
YES
NULL
Field
Type
Null
Key
Default
Extra
complianceRequired
tinyint(1)
YES
NULL
evidencesRequired
tinyint(1)
YES
NULL
solutionRequired
tinyint(1)
YES
NULL
trainingRequired
tinyint(1)
YES
NULL
witnessStatementRequired
tinyint(1)
YES
NULL
rootCauseAnalysisRequired
tinyint(1)
YES
NULL
capaRequired
tinyint(1)
YES
NULL
isCapaAssigned
tinyint(4)
NO
0
capaApprovedByRO
varchar(50)
YES
NULL
isCapaCompleted
tinyint(4)
NO
0
capaApprovedBySO
varchar(50)
YES
NULL
capaApprovedByMM
varchar(50)
YES
NULL
investigatedBy
int(11)
YES
MUL
NULL
reviewedBy
int(11)
YES
MUL
NULL
reviewedByMM
int(11)
YES
MUL
NULL
finalReviewedBy
int(11)
YES
MUL
NULL
capaCompletionDate
datetime
YES
NULL
capaCompletedBy
int(11)
YES
MUL
NULL
reviewedDate
datetime
YES
NULL
reviewedByMMDate
datetime
YES
NULL
finalReviewedByDate
datetime
YES
NULL
closedBy
int(11)
YES
MUL
NULL
workAreaId
int(11)
YES
MUL
NULL
stateMasterId
int(11)
YES
MUL
NULL
departmentId
int(11)
YES
MUL
NULL
otherDocumentRequired
tinyint(4)
NO
0
causeIndirect
varchar(255)
YES
NULL
causeImmediate
varchar(255)
YES
NULL
causeIndirectOther
varchar(255)
YES
NULL
causeImmediateOther
varchar(255)
YES
NULL
status
tinyint(4)
YES
0
createdBy
int(11)
YES
MUL
NULL
updatedBy
int(11)
YES
NULL
deleted
int(11)
YES
MUL
0
createdAt
datetime
NO
NULL
updatedAt
datetime
NO
NULL
reportEvidencesRequired
tinyint(1)
YES
NULL
injuryInfoRequired
tinyint(1)
YES
NULL
reviewEvidencesRequired
tinyint(1)
YES
NULL
shift
varchar(255)
YES
NULL
isBack
tinyint(4)
YES
0\nQuery - {user_query}'''
    )
    
    #response = requests.post(gemini_url, headers=headers, json={"prompt": prompt, "max_tokens": 150})
    response = model.generate_content(prompt)
    print(response.text.lstrip("```json").rstrip())
    try:
        parsed_response = json.loads(response.text.lstrip("```json").rstrip().rstrip("```"))
        return parsed_response
    except json.JSONDecodeError:
        print("Could not parse response to JSON")
    #return response.json()  # {"Query 1": "...", "Query 2": "..."}

# Step 6: Execute SQL Query
def execute_sql_query(connection, query):
    cursor = connection.cursor(dictionary=True)
    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()
    return results

# Step 7: Perform Similarity Search
def perform_similarity_search(faiss_index_path, id_mapping_path, query_text, model):
    # Load FAISS index and ID mapping
    index = faiss.read_index(faiss_index_path)
    with open(id_mapping_path, "rb") as f:
        id_mapping = pickle.load(f)

    # Generate embedding for query
    query_embedding = model.encode([query_text])
    
    # Perform search
    k = 5  # Top-k results
    distances, indices = index.search(query_embedding, k)
    
    return [id_mapping[idx] for idx in indices[0] if idx < len(id_mapping)]

# Step 8: Combine Results and Display
def search_and_filter(table_name, text_columns, user_query):
    connection = connect_to_db()

    # Extract and embed data if not already done
    data = extract_data(connection, table_name, text_columns)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings, id_mapping = generate_embeddings(data, text_columns, model)
    
    faiss_index_path = "incident_summary_index.faiss"
    id_mapping_path = "id_mapping.pkl"
    save_to_faiss(embeddings, id_mapping, faiss_index_path, id_mapping_path)

    # Query decomposition
    query_decomposition = decompose_query(text_columns, user_query)
    query_1 = query_decomposition["Query 1"]+ ";"
    query_2 = query_decomposition["Query 2"]
    print(query_1)

    # SQL filtering
    filtered_rows = execute_sql_query(connection, query_1)

    # Similarity search on filtered rows
    if filtered_rows:
        filtered_ids = [row['id'] for row in filtered_rows]
        faiss_ids = perform_similarity_search(faiss_index_path, id_mapping_path, query_2, model)

        # Intersection of results
        final_ids = set(filtered_ids).intersection(faiss_ids)
        
        # Fetch final rows
        if final_ids:
            #placeholders = ','.join(['%s'] * len(final_ids))
            placeholders = ','.join(map(str, final_ids))
            query = f"SELECT * FROM {table_name} WHERE id IN ({placeholders});"
            results = execute_sql_query(connection, query)
            for row in results:
                print(row)
        else:
            print("No rows match the similarity search.")
    else:
        print("No rows match the SQL query.")

    connection.close()

# Main Program
if __name__ == "__main__":
    table_name = "ehs_incident_details"
    text_columns = ["description", "summary"]
    user_query = "Filter incidents where incident type is nearMiss and a steering rod was broken."
    search_and_filter(table_name, text_columns, user_query)