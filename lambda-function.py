# this is the code for the lambda function that will call the SageMaker endpoint. Run in the AWS lambda console

import json
import boto3

ENDPOINT = "huggingface-pytorch-tgi-inference-2023-07-24-07-42-27-365" # this is the name of the endpoint created in SageMaker
runtime = boto3.client('runtime.sagemaker') # this is the boto3 client that can invoke the endpoint

# this is the lambda handler, which will be invoked by AWS Lambda when the endpoint is called
def lambda_handler(event, context):
    # TODO implement
    querty_params = event["queryStringParameters"]
    
    query = querty_params.get('query')
    
    if query is None:
        return {
            'statusCode': 400,
            'body': json.dumps('No query parameter provided')
        }
    
    #hyperparams
    payload =  {
        "inputs":query,
        "parameters": {
            "do_sample": True,
            "top_p":0.7,
            "temperature":0.3,
            "top_k":50,
            "max_new_tokens":512,
            "repetition_penalty":1.03
        }
    }
    
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT,ContentType="application/json",Body=json.dumps(payload))
    prediction = json.loads(response['Body'].read().decode('utf-8'))
    final_result = prediction[0]['generated_text']
    
    return {
        'statusCode': 200,
        'body': json.dumps(final_result)
    }