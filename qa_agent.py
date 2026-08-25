from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
import numpy as np

## reading the product overview markdown file 
with open("product_overview.md", "r", encoding="utf-8") as f:
    prod = f.read()

products = []

category = None
product_name = None
description = []


### reading the markdown file and extracting the product information
for line in prod.splitlines():


    ### checking if the line encountered is a category hader 
    if line.startswith("## ") and not line.startswith("### "):


        ##if its a category header, we will check if we have a product name and description save the previous before moving on to next
        if product_name is not None:
            products.append({
                "category": category,
                "product_name": product_name,
                "description": "\n".join(description).strip()
            })

        ## saving the current category name 
        category = line.replace("## ", "").strip()


        ## resetting product name and description to None for next 
        product_name = None
        description = []


    ### checking if the line encountered is a product header
    elif line.startswith("### "):

         ##if its a product header, we will check if we have a product name and description save the previous before moving on to next
        if product_name is not None:
            products.append({
                "category": category,
                "product_name": product_name,
                "description": "\n".join(description).strip()
            })

        ## saving the current product name and resetting description 
        product_name = line.replace("### ", "").strip()
        description = []

    else:
        ### if product name is not empty
        if product_name is not None:

            ## removing unnecessary lines like empty lines, images and page comments from the description
            if not line.strip():
                continue

            if line.strip().startswith("!["):
                continue

            if line.strip().startswith("<!-- page:"):
                continue


            ## appending description of current product to description list
            description.append(line.strip())


## adding all products to the products list after reading the entire markdown file
if product_name is not None:
    products.append({
        "category": category,
        "product_name": product_name,
        "description": "\n".join(description).strip()
    })

### creating a list of chunk text for each product to be used for embedding
chunk_text = []

### parsing through the products list and creating a chunk text for each product
for product in products:
    chunk_text.append(
        f"Category: {product['category']}\n"
        f"Product: {product['product_name']}\n"
        f"Description: {product['description']}\n\n"
    )

## creating embeddings for each chunk of text using the sentence-transformers model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embeddings = model.encode(chunk_text)


## initializing the Gemini client to use for generating responses
client = genai.Client()

## creating a Loop for continuous interaction with the user until they choose to exit
while True:

    ## the user query is taken as input and the embeddings for the query is generated using the same model used for generating embeddings for the product chunks
    user_query = input(
        "Hey there! I'm your virtual assistant!!\n\n"
        "What query do you have on your mind today?\n\n"
        "User: "
    )


    ## embedding the user query using the same model used for generating embeddings for the product chunks
    query_embedding = model.encode(user_query)

    ## reshaping the single query embedding from a 1D vector to a 2D array because cosine_similarity expects 2D inputs.

    query_embedding = np.reshape(query_embedding, (1, -1))

    ## using cosine similarity to find the most relevant product chunks based on the user query
    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )


    ## calculating cosine similarity between the query embedding and all product embeddings to obtain a similarity score for each chunk
    top_k = np.argpartition(
        similarities[0],
        -3
    )[-3:]


    ### Creating a list of the top k most relevant product chunks to be used as context for generating the response
    chunk_llm_feed = []

    for i in top_k:
        chunk_llm_feed.append(chunk_text[i])

    
    ### joining the top k most relevant product chunks to create a context for generating the response
    context = "\n".join(chunk_llm_feed)


    ## creating a prompt for the Gemini model to generate a response based on the context and user query
    prompt = f"""
You are a product catalogue assistant.

Use the context below to answer the question.

Context:
{context}

Question:
{user_query}

Give a complete answer to the question.
Mention the product name if the context contains it.
Explain the feature using the information given in the context.
Do not invent information.
If the context does not contain the answer, say "I don't know."
"""


    ## generating a response using the Gemini model based on the prompt created above
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    answer = response.text


    ### printing the answer generated by the Gemini model to the user
    print("\nBot:", answer)


    ## asking the user if they want to continue the chat or exit
    continue_chat = input(
        "Do you want to continue the chat? (yes/no): "
    )

    if continue_chat.lower() != "yes":
        print("Thank you for using me, Goodbye!!")
        break