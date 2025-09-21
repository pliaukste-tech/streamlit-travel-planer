================
CODE SNIPPETS
================
TITLE: Install Google GenAI SDK for Gemini API (Python)
DESCRIPTION: Demonstrates the correct `pip` command to install the Google GenAI SDK (`google-genai`) for Python, contrasting it with deprecated or incorrect package names.

SOURCE: https://github.com/googleapis/python-genai/blob/main/codegen_instructions.md#_snippet_0

LANGUAGE: Shell
CODE:
```
pip install google-generativeai
```

LANGUAGE: Shell
CODE:
```
pip install google-ai-generativelanguage
```

LANGUAGE: Shell
CODE:
```
pip install google-genai
```

--------------------------------

TITLE: Install Google Gen AI Python SDK
DESCRIPTION: This command installs the `google-genai` Python client library using pip, making it available for use in Python projects.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_0

LANGUAGE: shell
CODE:
```
pip install google-genai
```

--------------------------------

TITLE: Download File for GenAI Content Generation Example
DESCRIPTION: Provides a shell command to download a sample text file from Google Cloud Storage, which can then be used as input for content generation with the Gemini Developer API.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_10

LANGUAGE: Shell
CODE:
```
!wget -q https://storage.googleapis.com/generativeai-downloads/data/a11.txt
```

--------------------------------

TITLE: List Tuning Jobs Asynchronously with Python
DESCRIPTION: Provides examples for asynchronously listing tuning jobs using `client.aio.tunings.list`. This includes iterating through jobs and using an asynchronous pager to manage results across pages.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_63

LANGUAGE: python
CODE:
```
async for job in await client.aio.tunings.list(config={'page_size': 10}):
    print(job)
```

LANGUAGE: python
CODE:
```
async_pager = await client.aio.tunings.list(config={'page_size': 10})
print(async_pager.page_size)
print(async_pager[0])
await async_pager.next_page()
print(async_pager[0])
```

--------------------------------

TITLE: List Tuning Jobs Synchronously with Python
DESCRIPTION: Illustrates how to synchronously list tuning jobs using `client.tunings.list`. Examples include simple iteration and using a synchronous pager to access job details and navigate pages.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_62

LANGUAGE: python
CODE:
```
for job in client.tunings.list(config={'page_size': 10}):
    print(job)
```

LANGUAGE: python
CODE:
```
pager = client.tunings.list(config={'page_size': 10})
print(pager.page_size)
print(pager[0])
pager.next_page()
print(pager[0])
```

--------------------------------

TITLE: List Batch Jobs (Synchronous, Python)
DESCRIPTION: This example demonstrates how to list batch jobs synchronously using the `client.batches.list` method. It iterates through the jobs, printing each one. A `ListBatchJobsConfig` is used to specify pagination options like `page_size`.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_81

LANGUAGE: python
CODE:
```
from google.genai import types

    for job in client.batches.list(config=types.ListBatchJobsConfig(page_size=10)):
        print(job)
```

--------------------------------

TITLE: Example JSONL Format for Batch Prediction Requests
DESCRIPTION: This JSON Lines example illustrates the expected format for a file containing multiple requests for a batch prediction job. Each line is a JSON object with a unique key and a 'request' field containing the model input.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_76

LANGUAGE: json
CODE:
```
{"key":"request_1", "request": {"contents": [{"parts": [{"text": "Explain how AI works in a few words"}]}], "generation_config": {"response_modalities": ["TEXT"]}}}
{"key":"request_2", "request": {"contents": [{"parts": [{"text": "Explain how Crypto works in a few words"}]}]}}
```

--------------------------------

TITLE: List Available Generative AI Models (Python)
DESCRIPTION: Provides examples for synchronously listing available base generative AI models using the `client.models.list()` method, including basic iteration and pagination to retrieve models in batches. This is useful for discovering which models are accessible for your project.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_25

LANGUAGE: python
CODE:
```
for model in client.models.list():
    print(model)
```

LANGUAGE: python
CODE:
```
pager = client.models.list(config={'page_size': 10})
print(pager.page_size)
print(pager[0])
pager.next_page()
print(pager[0])
```

--------------------------------

TITLE: List Generative AI Tuning Jobs Synchronously with Python
DESCRIPTION: These examples show how to synchronously list active or completed tuning jobs. The first iterates through jobs, and the second demonstrates using a pager object to manage paginated results.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_72

LANGUAGE: python
CODE:
```
for job in client.tunings.list(config={'page_size': 10}):
    print(job)
```

LANGUAGE: python
CODE:
```
pager = client.tunings.list(config={'page_size': 10})
print(pager.page_size)
print(pager[0])
pager.next_page()
print(pager[0])
```

--------------------------------

TITLE: Create Batch Prediction Jobs in Python
DESCRIPTION: Demonstrates creating batch prediction jobs. Examples include specifying a BigQuery table or GCS file as the source, and creating a job with inlined requests directly within the client call.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_64

LANGUAGE: python
CODE:
```
job = client.batches.create(
    model='gemini-2.0-flash-001',
    src='bq://my-project.my-dataset.my-table'  # or gcs://my-bucket/my-file.jsonl
)
```

LANGUAGE: python
CODE:
```
batch_job = client.batches.create(
    model="gemini-2.0-flash",
    src=[{
      "contents": [{
        "parts": [{
          "text": "Hello!",
        }],
       "role": "user",
     }],
     "config:": {"response_modalities": ["text"]},
    }]
)
```

--------------------------------

TITLE: Upload files for content generation using Google GenAI Python
DESCRIPTION: For larger files, this example demonstrates using `client.files.upload` to upload a file before passing it to the content generation method. This approach is suitable for handling substantial media files efficiently.

SOURCE: https://github.com/googleapis/python-genai/blob/main/codegen_instructions.md#_snippet_10

LANGUAGE: python
CODE:
```
f = client.files.upload(file=img_path)

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[f, "can you describe this image?"]
)
```

--------------------------------

TITLE: Generate Videos with Python GenAI API
DESCRIPTION: This example demonstrates how to generate videos using the `client.models.generate_videos` method. It specifies the model, a text prompt, and configuration details like number of videos, FPS, and duration. The code then polls the operation until it's complete to retrieve and display the generated video.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_42

LANGUAGE: python
CODE:
```
from google.genai import types

# Create operation
operation = client.models.generate_videos(
    model='veo-2.0-generate-001',
    prompt='A neon hologram of a cat driving at top speed',
    config=types.GenerateVideosConfig(
        number_of_videos=1,
        fps=24,
        duration_seconds=5,
        enhance_prompt=True,
    ),
)

# Poll operation
while not operation.done:
    time.sleep(20)
    operation = client.operations.get(operation)

video = operation.result.generated_videos[0].video
video.show()
```

--------------------------------

TITLE: List Batch Jobs (Asynchronous, Python)
DESCRIPTION: This example shows how to list batch jobs asynchronously using `client.aio.batches.list`. It uses `async for` and `await` to handle the asynchronous iteration, suitable for non-blocking I/O operations in an `asyncio` environment.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_83

LANGUAGE: python
CODE:
```
from google.genai import types

    async for job in await client.aio.batches.list(
        config=types.ListBatchJobsConfig(page_size=10)
    ):
        print(job)
```

--------------------------------

TITLE: Generate Content with a Tuned Generative AI Model in Python
DESCRIPTION: This example demonstrates how to use a newly tuned model to generate content. It calls the `generate_content` method on the tuned model's endpoint and prints the generated text response.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_66

LANGUAGE: python
CODE:
```
response = client.models.generate_content(
    model=tuning_job.tuned_model.endpoint,
    contents='why is the sky blue?',
)

print(response.text)
```

--------------------------------

TITLE: Initiate Model Tuning with Google Generative AI (Python)
DESCRIPTION: This Python snippet begins the process of initiating a model tuning job for a generative AI model. It sets up the model and starts defining the training dataset using `types.TuningDataset` for supervised fine-tuning, specifically for Vertex AI.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_62

LANGUAGE: python
CODE:
```
from google.genai import types

model = 'gemini-2.0-flash-001'
training_dataset = types.TuningDataset(
```

--------------------------------

TITLE: Grounding Queries with Google Search in GenAI
DESCRIPTION: This example illustrates how to use Google Search as a tool to ground generative AI queries, providing up-to-date information from the web. It shows how to configure the `generate_content` call with a `google_search` tool and how to extract the search query and URLs used for grounding from the response.

SOURCE: https://github.com/googleapis/python-genai/blob/main/codegen_instructions.md#_snippet_22

LANGUAGE: python
CODE:
```
from google import genai

client = genai.Client()

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='What was the score of the latest Olympique Lyonais' game?',
    config={"tools": [{"google_search": {}}]}
)

# Response
print(f"Response:\n {response.text}")
# Search details
print(f"Search Query: {response.candidates[0].grounding_metadata.web_search_queries}")
# Urls used for grounding
print(f"Search Pages: {', '.join([site.web.title for site in response.candidates[0].grounding_metadata.grounding_chunks])}")
```

--------------------------------

TITLE: List and Paginate Tuned Models (Python)
DESCRIPTION: Lists available tuned models, excluding base models, with pagination. The first example iterates through all models, while the second demonstrates explicit pagination using the `pager` object to retrieve models page by page.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_59

LANGUAGE: python
CODE:
```
for model in client.models.list(config={'page_size': 10, 'query_base': False}}):
    print(model)
```

LANGUAGE: python
CODE:
```
pager = client.models.list(config={'page_size': 10, 'query_base': False})
print(pager.page_size)
print(pager[0])
pager.next_page()
print(pager[0])
```

--------------------------------

TITLE: Manually Declare Gemini Function and Tool for Function Calling (Python)
DESCRIPTION: This example demonstrates how to define a FunctionDeclaration and a Tool object, then pass the tool to the generate_content method. This setup allows the model to suggest function calls without automatically invoking them, providing control over the execution flow.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_29

LANGUAGE: python
CODE:
```
from google.genai import types

    function = types.FunctionDeclaration(
        name='get_current_weather',
        description='Get the current weather in a given location',
        parameters=types.Schema(
            type='OBJECT',
            properties={
                'location': types.Schema(
                    type='STRING',
                    description='The city and state, e.g. San Francisco, CA',
                ),
            },
            required=['location'],
        ),
    )

    tool = types.Tool(function_declarations=[function])

    response = client.models.generate_content(
        model='gemini-2.0-flash-001',
        contents='What is the weather like in Boston?',
        config=types.GenerateContentConfig(
            tools=[tool],
        ),
    )
    print(response.function_calls[0])
```

--------------------------------

TITLE: Specify Response Modality (Python)
DESCRIPTION: Notes the incorrect usage of `types.ResponseModality.TEXT` as a standalone example, implying it's part of an outdated enum or usage pattern.

SOURCE: https://github.com/googleapis/python-genai/blob/main/codegen_instructions.md#_snippet_6

LANGUAGE: Python
CODE:
```
types.ResponseModality.TEXT
```

--------------------------------

TITLE: List Tuned Generative AI Models Synchronously with Python
DESCRIPTION: These examples demonstrate how to retrieve a list of tuned Generative AI models. The first shows simple iteration, while the second illustrates using a pager object to navigate through results, including accessing page size and moving to the next page.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_69

LANGUAGE: python
CODE:
```
for model in client.models.list(config={'page_size': 10, 'query_base': False}}):
    print(model)
```

LANGUAGE: python
CODE:
```
pager = client.models.list(config={'page_size': 10, 'query_base': False})
print(pager.page_size)
print(pager[0])
pager.next_page()
print(pager[0])
```

--------------------------------

TITLE: Submit a model tuning job to Gemini API with Python
DESCRIPTION: This Python example initiates a supervised fine-tuning job using `client.tunings.tune`. It takes a base model, a prepared training dataset, and a `CreateTuningJobConfig` which includes parameters like epoch count and a display name for the resulting tuned model.

SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_13

LANGUAGE: python
CODE:
```
from google.genai import types

tuning_job = client.tunings.tune(
    base_model=model,
    training_dataset=training_dataset,
    config=types.CreateTuningJobConfig(
        epoch_count=1, tuned_model_display_name='test_dataset_examples model'
    ),
)
print(tuning_job)
```

--------------------------------

TITLE: Generate content with image input using Google GenAI Python
DESCRIPTION: This example shows how to handle multimodal inputs by passing a PIL Image object within the `contents` list. It opens an image file and sends it along with a text prompt to the model for analysis and response generation.

SOURCE: https://github.com/googleapis/python-genai/blob/main/codegen_instructions.md#_snippet_8

LANGUAGE: python
CODE:
```
from google import genai
from PIL import Image

client = genai.Client()

image = Image.open(img_path)

response = client.models.generate_content(
  model='gemini-2.5-flash',
  contents=[image, "explain that image"],
)

print(response.text) # The output often is markdown
```

--------------------------------

TITLE: Get Batch Job Details by Name (Python)
DESCRIPTION: This code retrieves the details of an existing batch job by its name using the `get` method. It's useful for checking the current state or other properties of a specific job.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_79

LANGUAGE: python
CODE:
```
# Get a job by name
    job = client.batches.get(name=job.name)

    job.state
```

--------------------------------

TITLE: Count Tokens Asynchronously (Python)
DESCRIPTION: Provides an example of asynchronously counting tokens for a text prompt using `client.aio.models.count_tokens`. This allows for non-blocking token calculation in asynchronous applications.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_44

LANGUAGE: python
CODE:
```
response = await client.aio.models.count_tokens(
    model='gemini-2.0-flash-001',
    contents='why is the sky blue?',
)
print(response)
```

--------------------------------

TITLE: Set system instructions for model behavior in Google GenAI Python
DESCRIPTION: This snippet illustrates how to guide the model's behavior using system instructions. It sets a `system_instruction` within `types.GenerateContentConfig` to define the model's persona or role for content generation.

SOURCE: https://github.com/googleapis/python-genai/blob/main/codegen_instructions.md#_snippet_13

LANGUAGE: python
CODE:
```
from google import genai
from google.genai import types

client = genai.Client()

config = types.GenerateContentConfig(
    system_instruction="You are a pirate",
)

response = client.models.generate_content(
    model='gemini-2.5-flash',
    config=config,
)

print(response.text)
```

--------------------------------

TITLE: Generate Content Using Cached Data (Python)
DESCRIPTION: This Python example demonstrates how to generate content using a pre-existing cache. It calls `client.models.generate_content` and specifies the `cached_content` in the configuration to leverage the cached data, improving efficiency for repeated queries.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_61

LANGUAGE: python
CODE:
```
from google.genai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='Summarize the pdfs',
    config=types.GenerateContentConfig(
        cached_content=cached_content.name,
    ),
)
print(response.text)
```

--------------------------------

TITLE: Copy Files from GCS with gsutil
DESCRIPTION: This example uses the `gsutil` command-line tool to copy PDF files from a Google Cloud Storage (GCS) bucket to the local directory. These files are then used as inputs for subsequent file management and caching operations within the GenAI API.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_47

LANGUAGE: shell
CODE:
```
gsutil cp gs://cloud-samples-data/generative-ai/pdf/2312.11805v3.pdf .
gsutil cp gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf .
```

--------------------------------

TITLE: Update a Retrieved Tuned Generative AI Model with Python
DESCRIPTION: This example demonstrates updating the properties of a tuned model that has been retrieved from a list or pager. It updates the model's display name and description using its name.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_71

LANGUAGE: python
CODE:
```
model = pager[0]

model = client.models.update(
    model=model.name,
    config=types.UpdateModelConfig(
        display_name='my tuned model', description='my tuned model description'
    ),
)

print(model)
```

--------------------------------

TITLE: Get and Monitor Batch Prediction Job Status in Python
DESCRIPTION: Shows how to retrieve a specific batch job by name and continuously monitor its state until completion. This involves polling the job status with a delay.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_65

LANGUAGE: python
CODE:
```
job = client.batches.get(name=job.name)
```

LANGUAGE: python
CODE:
```
job.state
```

LANGUAGE: python
CODE:
```
completed_states = set(
    [
        'JOB_STATE_SUCCEEDED',
        'JOB_STATE_FAILED',
        'JOB_STATE_CANCELLED',
        'JOB_STATE_PAUSED',
    ]
)

while job.state not in completed_states:
    print(job.state)
    job = client.batches.get(name=job.name)
    time.sleep(30)
```

LANGUAGE: python
CODE:
```
job
```

--------------------------------

TITLE: List tuned models asynchronously with Pager for Gemini API (Python)
DESCRIPTION: This asynchronous Python example uses an async pager to list tuned models, allowing for pagination in a non-blocking manner. It prints the page size, accesses items, and then uses `await async_pager.next_page()` to load subsequent results.

SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_21

LANGUAGE: python
CODE:
```
async_pager = await client.aio.models.list(config={'page_size': 10, 'query_base': False})
print(async_pager.page_size)
print(async_pager[0])
await async_pager.next_page()
print(async_pager[0])
```

--------------------------------

TITLE: List Batch Prediction Jobs Synchronously with Python
DESCRIPTION: Explains how to synchronously list batch prediction jobs using `client.batches.list`. Examples cover simple iteration and using a synchronous pager to manage results and access job details across pages.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_66

LANGUAGE: python
CODE:
```
from google.genai import types

for job in client.batches.list(config=types.ListBatchJobsConfig(page_size=10)):
    print(job)
```

LANGUAGE: python
CODE:
```
from google.genai import types

pager = client.batches.list(config=types.ListBatchJobsConfig(page_size=10))
print(pager.page_size)
print(pager[0])
pager.next_page()
print(pager[0])
```

--------------------------------

TITLE: Stream Content Asynchronously with Python GenAI Client
DESCRIPTION: This example illustrates how to perform asynchronous content generation with streaming. By combining async for with await client.aio.models.generate_content_stream, applications can efficiently process streamed responses in a non-blocking manner, ideal for high-concurrency scenarios.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_36

LANGUAGE: Python
CODE:
```
async for chunk in await client.aio.models.generate_content_stream(
    model='gemini-2.0-flash-001', contents='Tell me a story in 300 words.'
):
    print(chunk.text, end='')
```

--------------------------------

TITLE: Integrate Model Context Protocol (MCP) Server as a Tool in Google GenAI (Python)
DESCRIPTION: This experimental snippet outlines the initial setup for integrating a local Model Context Protocol (MCP) server directly as a tool within the Google GenAI Python client. It shows the necessary imports for `mcp` and `google.genai` to begin the integration process.

SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_4

LANGUAGE: python
CODE:
```
import os
import asyncio
from datetime import datetime
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from google import genai

client = genai.Client()
```

--------------------------------

TITLE: Delete a Batch Job in Python
DESCRIPTION: This code example illustrates how to delete a specific batch job using the `client.batches.delete` method in the Google GenAI Python SDK. It requires the `name` of the job resource to be deleted.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_69

LANGUAGE: Python
CODE:
```
# Delete the job resource
delete_job = client.batches.delete(name=job.name)

delete_job
```

--------------------------------

TITLE: List tuned models using Pager for Gemini API (Python)
DESCRIPTION: This Python example demonstrates listing tuned models using a pager object, which allows for navigating through paginated results. It prints the page size, accesses an element, and then explicitly retrieves the next page of results using `pager.next_page()`.

SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_19

LANGUAGE: python
CODE:
```
pager = client.models.list(config={'page_size': 10, 'query_base': False})
print(pager.page_size)
print(pager[0])
pager.next_page()
print(pager[0])
```

--------------------------------

TITLE: Generate Content with Advanced Typed Configuration (Python)
DESCRIPTION: This example illustrates how to use Pydantic types for `generate_content` parameters, specifically `types.GenerateContentConfig`. It showcases a comprehensive set of generation parameters including `temperature`, `top_p`, `top_k`, `candidate_count`, `seed`, `max_output_tokens`, `stop_sequences`, `presence_penalty`, and `frequency_penalty` to fine-tune model responses.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_20

LANGUAGE: Python
CODE:
```
from google.genai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents=types.Part.from_text(text='Why is the sky blue?'),
    config=types.GenerateContentConfig(
        temperature=0,
        top_p=0.95,
        top_k=20,
        candidate_count=1,
        seed=5,
        max_output_tokens=100,
        stop_sequences=['STOP!'],
        presence_penalty=0.0,
        frequency_penalty=0.0,
    ),
)

print(response.text)
```

--------------------------------

TITLE: Send Asynchronous Streaming Chat Message (Python)
DESCRIPTION: This Python example shows how to send asynchronous, streaming chat messages. It uses `async for` to iterate over response chunks as they become available, enabling efficient handling of long or real-time responses.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_54

LANGUAGE: python
CODE:
```
chat = client.aio.chats.create(model='gemini-2.0-flash-001')
async for chunk in await chat.send_message_stream('tell me a story'):
    print(chunk.text, end='')
```

--------------------------------

TITLE: Get Enum Response as JSON from Gemini API in Python
DESCRIPTION: Shows how to retrieve an enum response from the Google Gemini API formatted as a JSON string by setting `response_mime_type` to 'application/json' while still using an `Enum` class for the `response_schema`.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_36

LANGUAGE: python
CODE:
```
class InstrumentEnum(Enum):
    PERCUSSION = 'Percussion'
    STRING = 'String'
    WOODWIND = 'Woodwind'
    BRASS = 'Brass'
    KEYBOARD = 'Keyboard'

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='What instrument plays multiple notes at once?',
    config={
        'response_mime_type': 'application/json',
        'response_schema': InstrumentEnum,
    },
)
print(response.text)
```

--------------------------------

TITLE: Get Enum Response as Text from Gemini API in Python
DESCRIPTION: Demonstrates how to configure the Google Gemini API to return an enum value directly as plain text by setting `response_mime_type` to 'text/x.enum' and providing an `Enum` class as the `response_schema`.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_35

LANGUAGE: python
CODE:
```
from enum import Enum

class InstrumentEnum(Enum):
    PERCUSSION = 'Percussion'
    STRING = 'String'
    WOODWIND = 'Woodwind'
    BRASS = 'Brass'
    KEYBOARD = 'Keyboard'

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='What instrument plays multiple notes at once?',
    config={
        'response_mime_type': 'text/x.enum',
        'response_schema': InstrumentEnum,
    },
)
print(response.text)
```

--------------------------------

TITLE: Send Synchronous Streaming Chat Message (Python)
DESCRIPTION: This Python example illustrates sending synchronous, streaming chat messages to a Gemini model. It iterates through chunks of the response as they arrive, allowing for real-time processing or display of the generated text.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_52

LANGUAGE: python
CODE:
```
chat = client.chats.create(model='gemini-2.0-flash-001')
for chunk in chat.send_message_stream('tell me a story'):
    print(chunk.text, end='')
```

--------------------------------

TITLE: Manage Multi-Turn Conversations with Google GenAI Python Client
DESCRIPTION: This example illustrates how to use the `chats` service to maintain conversation history for multi-turn interactions. It shows how to create a chat session, send messages, and retrieve the full conversation history, enabling a continuous dialogue with the model.

SOURCE: https://github.com/googleapis/python-genai/blob/main/codegen_instructions.md#_snippet_16

LANGUAGE: python
CODE:
```
from google import genai

client = genai.Client()
chat = client.chats.create(model="gemini-2.5-flash")

response = chat.send_message("I have 2 dogs in my house.")
print(response.text)

response = chat.send_message("How many paws are in my house?")
print(response.text)

for message in chat.get_history():
    print(f'role - {message.role}',end=": ")
    print(message.parts[0].text)
```

--------------------------------

TITLE: Stream Text Content Synchronously with Python GenAI Client
DESCRIPTION: This example shows how to generate content from the Gemini model in a synchronous streaming fashion. The model's output is returned in chunks, allowing for real-time processing as the content is generated, rather than waiting for the entire response.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_33

LANGUAGE: Python
CODE:
```
for chunk in client.models.generate_content_stream(
    model='gemini-2.0-flash-001', contents='Tell me a story in 300 words.'
):
    print(chunk.text, end='')
```

--------------------------------

TITLE: Generate content using a tuned model with Gemini API (Python)
DESCRIPTION: This Python example demonstrates how to make a `generate_content` request using a fine-tuned model. It references the tuned model's specific endpoint, obtained from the completed tuning job, to ensure the request utilizes the custom-trained model for generation.

SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_16

LANGUAGE: python
CODE:
```
response = client.models.generate_content(
    model=tuning_job.tuned_model.endpoint,
    contents='why is the sky blue?',
)

print(response.text)
```

--------------------------------

TITLE: Delete File from Google Generative AI (Python)
DESCRIPTION: This Python example illustrates how to delete a file previously uploaded to the Google Generative AI service. It uploads a file for demonstration purposes and then calls `client.files.delete` to remove it, freeing up resources.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_58

LANGUAGE: python
CODE:
```
file3 = client.files.upload(file='2312.11805v3.pdf')

client.files.delete(name=file3.name)
```

--------------------------------

TITLE: Implement Function Calling (Tools) with Google GenAI Python Client
DESCRIPTION: This example shows how to enable the GenAI model to use external tools (functions) to retrieve information or perform actions. It defines a Python function, makes it available to the model, and demonstrates how the model can request to call this function to answer a query, enhancing its capabilities beyond its training data.

SOURCE: https://github.com/googleapis/python-genai/blob/main/codegen_instructions.md#_snippet_18

LANGUAGE: python
CODE:
```
from google import genai
from google.genai import types

client = genai.Client()

# Define a function that the model can call (to access external information)
def get_current_weather(city: str) -> str:
    """Returns the current weather in a given city. For this example, it's hardcoded."""
    if "boston" in city.lower():
        return "The weather in Boston is 15°C and sunny."
    else:
        return f"Weather data for {city} is not available."

# Make the function available to the model as a tool
response = client.models.generate_content(
  model='gemini-2.5-flash',
  contents="What is the weather like in Boston?",
  config=types.GenerateContentConfig(
      tools=[get_current_weather]
  ),
)
# The model may respond with a request to call the function
if response.function_calls:
    print("Function calls requested by the model:")
    for function_call in response.function_calls:
        print(f"- Function: {function_call.name}")
        print(f"- Args: {dict(function_call.args)}")
else:
    print("The model responded directly:")
    print(response.text)
```

--------------------------------

TITLE: Upscale Generated Image with Google Gemini API (Python)
DESCRIPTION: This example shows how to upscale a previously generated image using the `client.models.upscale_image` method. It takes the generated image as input and specifies an upscale factor (e.g., 'x2'). This functionality is currently only supported in Vertex AI and uses the `imagen-3.0-generate-002` model.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_40

LANGUAGE: python
CODE:
```
from google.genai import types

# Upscale the generated image from above
response2 = client.models.upscale_image(
    model='imagen-3.0-generate-002',
    image=response1.generated_images[0].image,
    upscale_factor='x2',
    config=types.UpscaleImageConfig(
        include_rai_reason=True,
        output_mime_type='image/jpeg',
    ),
)
response2.generated_images[0].image.show()
```

--------------------------------

TITLE: List Batch Jobs Asynchronously with Pager in Python
DESCRIPTION: This snippet demonstrates how to asynchronously list batch jobs using the `google.genai` library in Python. It utilizes a pager to retrieve results in chunks, allowing for efficient handling of large lists of jobs. The example shows how to configure the page size and iterate through pages.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_68

LANGUAGE: Python
CODE:
```
from google.genai import types

async_pager = await client.aio.batches.list(
    config=types.ListBatchJobsConfig(page_size=10)
)
print(async_pager.page_size)
print(async_pager[0])
await async_pager.next_page()
print(async_pager[0])
```

--------------------------------

TITLE: Create Batch Prediction Job with Inlined Requests in Python
DESCRIPTION: This example shows how to create a batch prediction job by providing requests directly within the API call, rather than from an external file. It defines the model and a list of content requests for batch processing.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_75

LANGUAGE: python
CODE:
```
# Create a batch job with inlined requests
batch_job = client.batches.create(
    model="gemini-2.0-flash",
    src=[{
      "contents": [{
        "parts": [{
          "text": "Hello!",
        }],
       "role": "user",
     }],
     "config:": {"response_modalities": ["text"]},
    }],
)
```

--------------------------------

TITLE: Adjust thinking budget for Gemini 2.5 models in Google GenAI Python
DESCRIPTION: This example demonstrates how to configure the `thinking_budget` for Gemini 2.5 series models using `types.ThinkingConfig`. Setting the budget to zero can reduce latency by turning off the model's 'thinking' process, where supported.

SOURCE: https://github.com/googleapis/python-genai/blob/main/codegen_instructions.md#_snippet_12

LANGUAGE: python
CODE:
```
from google import genai
from google.genai import types

client = genai.Client()

client.models.generate_content(
  model='gemini-2.5-flash',
  contents="What is AI?",
  config=types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(
      thinking_budget=0
    )
  )
)
```

--------------------------------

TITLE: Get a tuning job status from Gemini API with Python
DESCRIPTION: This Python snippet retrieves the current status and detailed information of an existing tuning job. It uses `client.tunings.get` with the tuning job's name to fetch its information, which is then printed to the console.

SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_14

LANGUAGE: python
CODE:
```
tuning_job = client.tunings.get(name=tuning_job.name)
print(tuning_job)
```

--------------------------------

TITLE: Send Synchronous Streaming Chat Messages with Python GenAI
DESCRIPTION: This example demonstrates sending chat messages synchronously with streaming enabled. It creates a chat session and then iterates over chunks of the response as they arrive, printing each chunk. This is useful for displaying responses incrementally to the user.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_44

LANGUAGE: python
CODE:
```
chat = client.chats.create(model='gemini-2.0-flash-001')
for chunk in chat.send_message_stream('tell me a story'):
    print(chunk.text, end='')  # end='' is optional, for demo purposes.
```

--------------------------------

TITLE: Delete Files from GenAI API with Python
DESCRIPTION: This example demonstrates how to delete an uploaded file from the GenAI API using `client.files.delete`. It first uploads a file and then uses its name to initiate the deletion. This helps in managing storage and cleaning up unused files.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_50

LANGUAGE: python
CODE:
```
file3 = client.files.upload(file='2312.11805v3.pdf')

client.files.delete(name=file3.name)
```

--------------------------------

TITLE: Initialize Gemini API Client (Python)
DESCRIPTION: Shows the correct way to initialize the Gemini API client using `genai.Client()` from the Google GenAI SDK, replacing the deprecated `genai.configure()` method.

SOURCE: https://github.com/googleapis/python-genai/blob/main/codegen_instructions.md#_snippet_2

LANGUAGE: Python
CODE:
```
genai.configure(api_key=...)
```

LANGUAGE: Python
CODE:
```
client = genai.Client(api_key="...")
```

--------------------------------

TITLE: Stream Image Content Synchronously with Python GenAI Client
DESCRIPTION: These examples illustrate how to stream content generation when the input includes image data. It covers two methods: loading images from a Google Cloud Storage URI using types.Part.from_uri and loading images from a local file system as bytes using types.Part.from_bytes.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_34

LANGUAGE: Python
CODE:
```
from google.genai import types

for chunk in client.models.generate_content_stream(
    model='gemini-2.0-flash-001',
    contents=[
        'What is this image about?',
        types.Part.from_uri(
            file_uri='gs://generativeai-downloads/images/scones.jpg',
            mime_type='image/jpeg',
        ),
    ],
):
    print(chunk.text, end='')
```

LANGUAGE: Python
CODE:
```
from google.genai import types

YOUR_IMAGE_PATH = 'your_image_path'
YOUR_IMAGE_MIME_TYPE = 'your_image_mime_type'
with open(YOUR_IMAGE_PATH, 'rb') as f:
    image_bytes = f.read()

for chunk in client.models.generate_content_stream(
    model='gemini-2.0-flash-001',
    contents=[
        'What is this image about?',
        types.Part.from_bytes(data=image_bytes, mime_type=YOUR_IMAGE_MIME_TYPE),
    ],
):
    print(chunk.text, end='')
```

--------------------------------

TITLE: Get file information from Google Gemini API with Python
DESCRIPTION: This Python code retrieves metadata for an already uploaded file using its assigned name. It demonstrates how to first upload a file to ensure it exists, then uses `client.files.get` with the file's unique name to fetch its details.

SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_7

LANGUAGE: python
CODE:
```
file1 = client.files.upload(file='2312.11805v3.pdf')
file_info = client.files.get(name=file1.name)
```

--------------------------------

TITLE: Apply Safety Settings to Content Generation (Python)
DESCRIPTION: This example shows how to configure safety settings for content generation using `types.SafetySetting`. It demonstrates blocking content that falls into a specific harm category (e.g., `HARM_CATEGORY_HATE_SPEECH`) at a defined threshold (`BLOCK_ONLY_HIGH`) to ensure responsible AI output.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_22

LANGUAGE: Python
CODE:
```
from google.genai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='Say something bad.',
    config=types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(
                category='HARM_CATEGORY_HATE_SPEECH',
                threshold='BLOCK_ONLY_HIGH',
            )
        ]
    ),
)
print(response.text)
```

--------------------------------

TITLE: Download a file for content generation
DESCRIPTION: Shows how to download a sample text file (`a11.txt`) from a Google Cloud Storage bucket using the `wget` command in a console environment. This file can then be used as input for content generation.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_13

LANGUAGE: console
CODE:
```
!wget -q https://storage.googleapis.com/generativeai-downloads/data/a11.txt
```

--------------------------------

TITLE: Configure safety settings for content generation in Google GenAI Python
DESCRIPTION: This example demonstrates how to apply safety configurations to content generation requests. It sets `safety_settings` within `types.GenerateContentConfig` to specify harm categories and block thresholds, ensuring generated content adheres to safety guidelines.

SOURCE: https://github.com/googleapis/python-genai/blob/main/codegen_instructions.md#_snippet_14

LANGUAGE: python
CODE:
```
from google import genai
from google.genai import types

client = genai.Client()

img = Image.open("/path/to/img")
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=['Do these look store-bought or homemade?', img],
    config=types.GenerateContentConfig(
      safety_settings=[
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        ),
      ]
    )
)

print(response.text)
```

--------------------------------

TITLE: Configure Model Generation with System Instructions and Parameters (Python)
DESCRIPTION: This snippet demonstrates how to use `generate_content` with various configuration parameters like `system_instruction`, `max_output_tokens`, and `temperature`. These settings influence the model's behavior and output characteristics. It shows how to set a specific system instruction for the model and control response length and randomness.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_19

LANGUAGE: Python
CODE:
```
from google.genai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='high',
    config=types.GenerateContentConfig(
        system_instruction='I say high, you say low',
        max_output_tokens=3,
        temperature=0.3,
    ),
)
print(response.text)
```

--------------------------------

TITLE: Count and Compute Tokens with Python GenAI Client
DESCRIPTION: This section provides examples for managing token usage. It shows how to synchronously count tokens in a given text input and how to compute tokens, a feature specifically supported in Vertex AI. An asynchronous version for counting tokens is also included for non-blocking operations.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_37

LANGUAGE: Python
CODE:
```
response = client.models.count_tokens(
    model='gemini-2.0-flash-001',
    contents='why is the sky blue?',
)
print(response)
```

LANGUAGE: Python
CODE:
```
response = client.models.compute_tokens(
    model='gemini-2.0-flash-001',
    contents='why is the sky blue?',
)
print(response)
```

LANGUAGE: Python
CODE:
```
response = await client.aio.models.count_tokens(
    model='gemini-2.0-flash-001',
    contents='why is the sky blue?',
)
print(response)
```

--------------------------------

TITLE: Get cached content details from Gemini API with Python
DESCRIPTION: This Python code retrieves information about a previously created cached content entry. It uses `client.caches.get` and the unique name of the cached content to fetch its details, allowing for verification or further processing.

SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_10

LANGUAGE: python
CODE:
```
cached_content = client.caches.get(name=cached_content.name)
```

--------------------------------

TITLE: Create Google Gen AI Client Using Environment Variables
DESCRIPTION: This Python code initializes a `genai.Client` instance without explicit parameters, relying on previously set environment variables (`GOOGLE_API_KEY` or Vertex AI related variables) for configuration and authentication.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_6

LANGUAGE: python
CODE:
```
from google import genai

client = genai.Client()
```

--------------------------------

TITLE: Delete a file from Google Gemini API with Python
DESCRIPTION: This Python example demonstrates how to delete a file from the Gemini Developer API. It first uploads a file to ensure there's a target, and then uses `client.files.delete` with the file's name to remove it from the service.

SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_8

LANGUAGE: python
CODE:
```
file3 = client.files.upload(file='2312.11805v3.pdf')

client.files.delete(name=file3.name)
```

--------------------------------

TITLE: Embed Content with Python GenAI Client
DESCRIPTION: These examples demonstrate how to generate embeddings for text content using the Google Generative AI client. It covers embedding a single piece of text and embedding multiple pieces of text while applying configuration options, such as specifying output dimensionality.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_38

LANGUAGE: Python
CODE:
```
response = client.models.embed_content(
    model='text-embedding-004',
    contents='why is the sky blue?',
)
print(response)
```

LANGUAGE: Python
CODE:
```
from google.genai import types

# multiple contents with config
response = client.models.embed_content(
    model='text-embedding-004',
    contents=['why is the sky blue?', 'What is your age?'],
    config=types.EmbedContentConfig(output_dimensionality=10),
)

print(response)
```

--------------------------------

TITLE: Get details of a tuned model from Gemini API with Python
DESCRIPTION: This Python code retrieves detailed information about a specific fine-tuned model. It uses `client.models.get` with the model's identifier, which is typically obtained from a completed tuning job, and prints the full model object.

SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_17

LANGUAGE: python
CODE:
```
tuned_model = client.models.get(model=tuning_job.tuned_model.model)
print(tuned_model)
```

--------------------------------

TITLE: Manually Declare a Function as a Tool for Google Gemini in Python
DESCRIPTION: This example demonstrates how to manually declare a function using `types.FunctionDeclaration` and `types.Tool` to be passed to the Google Gemini API. This approach provides more control over function definitions and allows the model to return a function call part in the response, which can then be invoked manually.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_25

LANGUAGE: Python
CODE:
```
from google.genai import types

function = types.FunctionDeclaration(
    name='get_current_weather',
    description='Get the current weather in a given location',
    parameters=types.Schema(
        type='OBJECT',
        properties={
            'location': types.Schema(
                type='STRING',
                description='The city and state, e.g. San Francisco, CA',
            ),
        },
        required=['location'],
    ),
)

tool = types.Tool(function_declarations=[function])

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='What is the weather like in Boston?',
    config=types.GenerateContentConfig(
        tools=[tool],
    ),
)
print(response.function_calls[0])
```

--------------------------------

TITLE: List Available Base Models (Python)
DESCRIPTION: This snippet demonstrates how to list available base models, covering both synchronous and asynchronous approaches. It shows iterating through all models and using pagination to retrieve models in batches, accessing page properties and navigating to the next page for both paradigms.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_21

LANGUAGE: Python
CODE:
```
for model in client.models.list():
    print(model)

pager = client.models.list(config={'page_size': 10})
print(pager.page_size)
print(pager[0])
pager.next_page()
print(pager[0])
```

LANGUAGE: Python
CODE:
```
async for job in await client.aio.models.list():
    print(job)

async_pager = await client.aio.models.list(config={'page_size': 10})
print(async_pager.page_size)
print(async_pager[0])
await async_pager.next_page()
print(async_pager[0])
```

--------------------------------

TITLE: Configure Generative AI Model Parameters (Python)
DESCRIPTION: Demonstrates how to set various configuration parameters like system instructions, max output tokens, and temperature when generating content with the Google Generative AI SDK. These settings allow fine-grained control over the model's response behavior.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_23

LANGUAGE: python
CODE:
```
from google.genai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='high',
    config=types.GenerateContentConfig(
        system_instruction='I say high, you say low',
        max_output_tokens=3,
        temperature=0.3,
    ),
)
print(response.text)
```

--------------------------------

TITLE: List Batch Prediction Jobs Synchronously using Python
DESCRIPTION: This snippet demonstrates how to list batch prediction jobs. It includes examples for iterating through jobs directly and using a synchronous pager object to manage results, access page size, and navigate through available pages.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_8

LANGUAGE: python
CODE:
```
from google.genai import types

for job in client.batches.list(config=types.ListBatchJobsConfig(page_size=10)):
    print(job)
```

LANGUAGE: python
CODE:
```
from google.genai import types

pager = client.batches.list(config=types.ListBatchJobsConfig(page_size=10))
print(pager.page_size)
print(pager[0])
pager.next_page()
print(pager[0])
```

--------------------------------

TITLE: Invoke a Function Call and Pass Response to Google GenAI Model (Python)
DESCRIPTION: This example illustrates how to extract function call details from a model's response, execute the corresponding Python function (e.g., `get_current_weather`), and handle potential exceptions. The function's result is then formatted as a `types.Part.from_function_response` and sent back to the model as part of the conversation history for further processing.

SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_1

LANGUAGE: python
CODE:
```
from google.genai import types

user_prompt_content = types.Content(
    role='user',
    parts=[types.Part.from_text(text='What is the weather like in Boston?')],
)
function_call_part = response.function_calls[0]
function_call_content = response.candidates[0].content


try:
    function_result = get_current_weather(
        **function_call_part.function_call.args
    )
    function_response = {'result': function_result}
except (
    Exception
) as e:  # instead of raising the exception, you can let the model handle it
    function_response = {'error': str(e)}


function_response_part = types.Part.from_function_response(
    name=function_call_part.name,
    response=function_response,
)
function_response_content = types.Content(
    role='tool', parts=[function_response_part]
)

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents=[
        user_prompt_content,
        function_call_content,
        function_response_content,
    ],
    config=types.GenerateContentConfig(
        tools=[tool],
    ),
)

print(response.text)
```

--------------------------------

TITLE: Create Google Gen AI Client for Vertex AI API
DESCRIPTION: This Python code initializes a `genai.Client` instance configured for the Vertex AI API, specifying `vertexai=True`, `project`, and `location`. This client enables interaction with Google's generative models hosted on Vertex AI.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_3

LANGUAGE: python
CODE:
```
from google import genai

# Only run this block for Vertex AI API
client = genai.Client(
    vertexai=True, project='your-project-id', location='us-central1'
)
```

--------------------------------

TITLE: Define Gemini response schema using a dictionary in Python
DESCRIPTION: This example shows an alternative method to define a structured JSON response schema for the Gemini model using a Python dictionary. This dictionary directly specifies the `required` fields, `properties` with their types (e.g., 'STRING', 'INTEGER'), and the overall `type` as 'OBJECT' for the expected JSON output.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_30

LANGUAGE: Python
CODE:
```
from google.genai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='Give me information for the United States.',
    config=types.GenerateContentConfig(
        response_mime_type='application/json',
        response_schema={
            'required': [
                'name',
                'population',
                'capital',
                'continent',
                'gdp',
                'official_language',
                'total_area_sq_mi',
            ],
            'properties': {
                'name': {'type': 'STRING'},
                'population': {'type': 'INTEGER'},
                'capital': {'type': 'STRING'},
                'continent': {'type': 'STRING'},
                'gdp': {'type': 'INTEGER'},
                'official_language': {'type': 'STRING'},
                'total_area_sq_mi': {'type': 'INTEGER'},
            },
            'type': 'OBJECT',
        },
    ),
)
print(response.text)
```

--------------------------------

TITLE: Generate content using cached data in Gemini API with Python
DESCRIPTION: This Python example shows how to use cached content when generating responses with the Gemini API. It calls `client.models.generate_content`, specifying the model, a prompt, and the `cached_content` name in the configuration to leverage pre-processed or stored data for generation.

SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_11

LANGUAGE: python
CODE:
```
from google.genai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='Summarize the pdfs',
    config=types.GenerateContentConfig(
        cached_content=cached_content.name,
    ),
)
print(response.text)
```

--------------------------------

TITLE: Manually Invoke Function Call and Pass Response to Google Gemini in Python
DESCRIPTION: This example illustrates the process of manually invoking a function after receiving a function call part from the Google Gemini model. It shows how to handle the function's execution, capture its result (or an error), and then format this response to be passed back to the model for continued conversation or processing.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_26

LANGUAGE: Python
CODE:
```
from google.genai import types

user_prompt_content = types.Content(
    role='user',
    parts=[types.Part.from_text(text='What is the weather like in Boston?')],
)
function_call_part = response.function_calls[0]
function_call_content = response.candidates[0].content

try:
    function_result = get_current_weather(
        **function_call_part.function_call.args
    )
    function_response = {'result': function_result}
except (
    Exception
) as e:  # instead of raising the exception, you can let the model handle it
    function_response = {'error': str(e)}

function_response_part = types.Part.from_function_response(
    name=function_call_part.name,
    response=function_response,
)
function_response_content = types.Content(
    role='tool', parts=[function_response_part]
)

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents=[
        user_prompt_content,
        function_call_content,
        function_response_content,
    ],
    config=types.GenerateContentConfig(
        tools=[tool],
    ),
)

print(response.text)
```

--------------------------------

TITLE: Generate Videos with Google Generative AI Python Client
DESCRIPTION: This Python code demonstrates how to generate videos using the `generate_videos` method of the client. It specifies the model, a creative prompt, and video configuration parameters like duration and frame rate. The snippet also includes polling the operation until the video generation is complete and then displaying the result.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_50

LANGUAGE: python
CODE:
```
operation = client.models.generate_videos(
    model='veo-2.0-generate-001',
    prompt='A neon hologram of a cat driving at top speed',
    config=types.GenerateVideosConfig(
        number_of_videos=1,
        fps=24,
        duration_seconds=5,
        enhance_prompt=True,
    ),
)

# Poll operation
while not operation.done:
    time.sleep(20)
    operation = client.operations.get(operation)

video = operation.result.generated_videos[0].video
video.show()
```

--------------------------------

TITLE: List Tuned Models Asynchronously with Python
DESCRIPTION: Demonstrates how to asynchronously list tuned models using the `client.aio.models.list` method, including iterating through results and using an asynchronous pager to access model details and navigate pages.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_60

LANGUAGE: python
CODE:
```
async for job in await client.aio.models.list(config={'page_size': 10, 'query_base': False}):
    print(job)
```

LANGUAGE: python
CODE:
```
async_pager = await client.aio.models.list(config={'page_size': 10, 'query_base': False})
print(async_pager.page_size)
print(async_pager[0])
await async_pager.next_page()
print(async_pager[0])
```

--------------------------------

TITLE: Configure maximum automatic function calls in Gemini with Python
DESCRIPTION: This example illustrates how to limit the number of automatic function call turns by setting `maximum_remote_calls` within `AutomaticFunctionCallingConfig`. It shows how to allow one turn for automatic function calling by setting `maximum_remote_calls=2` (which means 1 initial call + 1 follow-up call).

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_28

LANGUAGE: Python
CODE:
```
from google.genai import types

def get_current_weather(location: str) -> str:
    """Returns the current weather.

    Args:
        location: The city and state, e.g. San Francisco, CA
    """
    return "sunny"

response = client.models.generate_content(
    model="gemini-2.0-flash-001",
    contents="What is the weather like in Boston?",
    config=types.GenerateContentConfig(
        tools=[get_current_weather],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            maximum_remote_calls=2
        ),
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode='ANY')
        ),
    ),
)
```

--------------------------------

TITLE: Edit Images with Gemini Native Image Generation Model in Google GenAI Python Client
DESCRIPTION: This example shows how to edit images using the Gemini native image generation model, specifically in chat mode. It demonstrates sending an image along with a text prompt to modify it, and then processing the generated image output, allowing for iterative image manipulation.

SOURCE: https://github.com/googleapis/python-genai/blob/main/codegen_instructions.md#_snippet_20

LANGUAGE: python
CODE:
```
from google import genai
from PIL import Image
from io import BytesIO

client = genai.Client()

prompt = """
  Create a picture of my cat eating a nano-banana in a fancy restaurant under the gemini constellation
"""
image = PIL.Image.open('/path/to/image.png')

# Create the chat
chat = client.chats.create(model="gemini-2.5-flash-image-preview")
# Send the image and ask for it to be edited
response = chat.send_message([prompt, image])

# Get the text and the image generated
for i, part in enumerate(response.candidates[0].content.parts):
  if part.text is not None:
    print(part.text)
  elif part.inline_data is not None:
    image = Image.open(BytesIO(part.inline_data.data))
    image.save(f"generated_image_{i}.png") # Multiple images can be generated

# Continue iterating
chat.send_message("Can you make it a bananas foster?")
```

--------------------------------

TITLE: Disable Automatic Function Calling in Python with Google Gemini
DESCRIPTION: This example shows how to disable the automatic function calling feature when passing a Python function as a tool to the Google Gemini API. When disabled, the model will return a list of function call parts in the response instead of automatically invoking the function.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_24

LANGUAGE: Python
CODE:
```
from google.genai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='What is the weather like in Boston?',
    config=types.GenerateContentConfig(
        tools=[get_current_weather],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
        ),
    ),
)
```

--------------------------------

TITLE: Understanding Content and Part Hierarchy in GenAI
DESCRIPTION: This section explains the `Content` and `Part` objects, which are fundamental building blocks for the `generate_content` API. It contrasts a simpler, shorthand API call with a more explicit structure using `types.Content` and `types.Part.from_text`, demonstrating how to achieve the same result with different levels of control over the input structure.

SOURCE: https://github.com/googleapis/python-genai/blob/main/codegen_instructions.md#_snippet_23

LANGUAGE: python
CODE:
```
from google import genai

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="How does AI work?"
)
print(response.text)
```

LANGUAGE: python
CODE:
```
from google import genai
from google.genai import types

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
      types.Content(role="user", parts=[types.Part.from_text(text="How does AI work?")]),
    ]
)
print(response.text)
```

--------------------------------

TITLE: Limit Automatic Function Calling Turns with ANY Mode in Google GenAI (Python)
DESCRIPTION: This example demonstrates how to control the number of automatic function calling turns when using 'ANY' mode. By configuring `automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=N)`, you can specify how many times the SDK should automatically invoke registered Python functions before returning control to the user.

SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_3

LANGUAGE: python
CODE:
```
from google.genai import types

def get_current_weather(location: str) -> str:
    """Returns the current weather.

    Args:
      location: The city and state, e.g. San Francisco, CA
    """
    return "sunny"

response = client.models.generate_content(
    model="gemini-2.0-flash-001",
    contents="What is the weather like in Boston?",
    config=types.GenerateContentConfig(
        tools=[get_current_weather],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            maximum_remote_calls=2
        ),
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode='ANY')
        ),
    ),
)
```

--------------------------------

TITLE: Manually Declare and Pass a Function as a Tool in Google GenAI Python
DESCRIPTION: This Python example illustrates how to manually declare a function using `types.FunctionDeclaration` and encapsulate it within a `types.Tool` object. This tool is then passed to the `generate_content` method. The model will recognize this declared function and return a function call part in its response when appropriate, rather than executing it automatically.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_1

LANGUAGE: python
CODE:
```
from google.genai import types

function = types.FunctionDeclaration(
    name='get_current_weather',
    description='Get the current weather in a given location',
    parameters=types.Schema(
        type='OBJECT',
        properties={
            'location': types.Schema(
                type='STRING',
                description='The city and state, e.g. San Francisco, CA',
            ),
        },
        required=['location'],
    ),
)

tool = types.Tool(function_declarations=[function])

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='What is the weather like in Boston?',
    config=types.GenerateContentConfig(
        tools=[tool],
    ),
)
print(response.function_calls[0])
```

--------------------------------

TITLE: Generate Videos with Google GenAI Python Client
DESCRIPTION: This snippet demonstrates how to generate videos using the Veo models via the Google Generative AI Python client. It includes steps for setting up the client, specifying model and prompt, handling optional image input, configuring video parameters like aspect ratio and duration, and saving the generated video files. Users should be aware of potential costs associated with Veo model usage.

SOURCE: https://github.com/googleapis/python-genai/blob/main/codegen_instructions.md#_snippet_21

LANGUAGE: python
CODE:
```
import time
from google import genai
from google.genai import types
from PIL import Image

client = genai.Client()

PIL_image = Image.open("path/to/image.png") # Optional

operation = client.models.generate_videos(
    model="veo-3.0-fast-generate-preview",
    prompt="Panning wide shot of a calico kitten sleeping in the sunshine",
    image = PIL_image,
    config=types.GenerateVideosConfig(
        person_generation="dont_allow",  # "dont_allow" or "allow_adult"
        aspect_ratio="16:9",  # "16:9" or "9:16"
        number_of_videos=1, # supported value is 1-4, use 1 by default
        duration_seconds=8, # supported value is 5-8
    ),
)

while not operation.done:
    time.sleep(20)
    operation = client.operations.get(operation)

for n, generated_video in enumerate(operation.response.generated_videos):
    client.files.download(file=generated_video.video) # just file=, no need for path= as it doesn't save yet
    generated_video.video.save(f"video{n}.mp4")  # saves the video
```

--------------------------------

TITLE: Generate text from a prompt using Google GenAI Python
DESCRIPTION: This snippet demonstrates how to perform basic text generation using the `google-genai` library. It initializes a client, specifies a model, and sends a text prompt to generate a response, which is then printed.

SOURCE: https://github.com/googleapis/python-genai/blob/main/codegen_instructions.md#_snippet_7

LANGUAGE: python
CODE:
```
from google import genai

client = genai.Client()

response = client.models.generate_content(
  model='gemini-2.5-flash',
  contents='why is the sky blue?',
)

print(response.text) # output is often markdown
```

--------------------------------

TITLE: Edit Image with Google Gemini API (Python)
DESCRIPTION: This snippet demonstrates how to edit an image using the `client.models.edit_image` method. It utilizes `RawReferenceImage` to provide the base image and `MaskReferenceImage` to define areas for editing (e.g., masking the background). The `imagen-3.0-capability-001` model is used, and the example sets an edit mode for inpainting/insertion. Image editing is currently only supported in Vertex AI.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_41

LANGUAGE: python
CODE:
```
# Edit the generated image from above
from google.genai import types
from google.genai.types import RawReferenceImage, MaskReferenceImage

raw_ref_image = RawReferenceImage(
    reference_id=1,
    reference_image=response1.generated_images[0].image,
)

# Model computes a mask of the background
mask_ref_image = MaskReferenceImage(
    reference_id=2,
    config=types.MaskReferenceConfig(
        mask_mode='MASK_MODE_BACKGROUND',
        mask_dilation=0,
    ),
)

response3 = client.models.edit_image(
    model='imagen-3.0-capability-001',
    prompt='Sunlight and clear sky',
    reference_images=[raw_ref_image, mask_ref_image],
    config=types.EditImageConfig(
        edit_mode='EDIT_MODE_INPAINT_INSERTION',
        number_of_images=1,
        include_rai_reason=True,
        output_mime_type='image/jpeg',
    ),
)
response3.generated_images[0].image.show()
```

--------------------------------

TITLE: Create Google Gen AI Client for Gemini Developer API
DESCRIPTION: This Python code initializes a `genai.Client` instance specifically for the Gemini Developer API, requiring an `api_key` for authentication. This client is used to interact with Google's generative models via the Gemini Developer API.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_2

LANGUAGE: python
CODE:
```
from google import genai

# Only run this block for Gemini Developer API
client = genai.Client(api_key='GEMINI_API_KEY')
```

--------------------------------

TITLE: Disable Automatic Function Calling in Gemini ANY Tool Config Mode (Python)
DESCRIPTION: This example demonstrates how to explicitly disable the SDK's automatic function calling mechanism when the tool_config mode is set to ANY. By setting automatic_function_calling.disable=True, you ensure that the model's response will always contain function call parts, requiring manual handling of function invocation.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_31

LANGUAGE: python
CODE:
```
from google.genai import types

    def get_current_weather(location: str) -> str:
        """Returns the current weather.

        Args:
            location: The city and state, e.g. San Francisco, CA
        """
        return "sunny"

    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents="What is the weather like in Boston?",
        config=types.GenerateContentConfig(
            tools=[get_current_weather],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode='ANY')
            ),
        ),
    )
```

--------------------------------

TITLE: Import Gemini API Client and Types (Python)
DESCRIPTION: Illustrates the correct Python import statements for the Google GenAI SDK (`google.genai`) and its types, differentiating from legacy `google.generativeai` or `google.ai` imports.

SOURCE: https://github.com/googleapis/python-genai/blob/main/codegen_instructions.md#_snippet_1

LANGUAGE: Python
CODE:
```
import google.generativeai as genai
```

LANGUAGE: Python
CODE:
```
from google import genai
```

LANGUAGE: Python
CODE:
```
from google.ai import generativelanguage_v1
```

LANGUAGE: Python
CODE:
```
from google import genai
```

LANGUAGE: Python
CODE:
```
from google.generativeai
```

LANGUAGE: Python
CODE:
```
from google import genai
```

LANGUAGE: Python
CODE:
```
from google.generativeai import types
```

LANGUAGE: Python
CODE:
```
from google.genai import types
```

--------------------------------

TITLE: Copy files to local environment using gsutil command
DESCRIPTION: This command-line snippet demonstrates how to copy PDF files from a Google Cloud Storage bucket to the current local directory using `gsutil`. These files are then used as local inputs for subsequent Gemini API operations.

SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_5

LANGUAGE: cmd
CODE:
```
!gsutil cp gs://cloud-samples-data/generative-ai/pdf/2312.11805v3.pdf .
!gsutil cp gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf .
```

--------------------------------

TITLE: Configure Async Client Options for Google Gen AI (Aiohttp)
DESCRIPTION: This Python code demonstrates how to pass additional arguments to the underlying `aiohttp.ClientSession.request()` method via `http_options.async_client_args` when using the `aiohttp` async client, allowing for fine-grained control over async requests.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_9

LANGUAGE: python
CODE:
```
http_options = types.HttpOptions(
    async_client_args={'cookies': ..., 'ssl': ...},
)

client=Client(..., http_options=http_options)
```

--------------------------------

TITLE: Generate Content with Gemini API (Python)
DESCRIPTION: Compares the correct method for generating content using `client.models.generate_content()` with the deprecated `model.generate_content()` from older SDK versions. Also covers streaming.

SOURCE: https://github.com/googleapis/python-genai/blob/main/codegen_instructions.md#_snippet_3

LANGUAGE: Python
CODE:
```
model = genai.GenerativeModel(...)
```

LANGUAGE: Python
CODE:
```
model.generate_content(...)
```

LANGUAGE: Python
CODE:
```
client.models.generate_content(...)
```

LANGUAGE: Python
CODE:
```
response = model.generate_content(..., stream=True)
```

LANGUAGE: Python
CODE:
```
client.models.generate_content_stream(...)
```

--------------------------------

TITLE: Create Batch Job with Google GenAI Python SDK
DESCRIPTION: This snippet demonstrates how to create a new batch job using the Google GenAI Python SDK. It specifies the model to use and the source file for the batch processing. The `create` method returns a `BatchJob` object.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_78

LANGUAGE: python
CODE:
```
batch_job = client.batches.create(
        model="gemini-2.0-flash",
        src="files/file_name",
    )
```

--------------------------------

TITLE: Asynchronously List Available Generative AI Models (Python)
DESCRIPTION: Demonstrates how to asynchronously list available base generative AI models using `client.aio.models.list()`, including basic asynchronous iteration and handling pagination for efficient retrieval. Asynchronous operations are beneficial for non-blocking applications.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_26

LANGUAGE: python
CODE:
```
async for job in await client.aio.models.list():
    print(job)
```

LANGUAGE: python
CODE:
```
async_pager = await client.aio.models.list(config={'page_size': 10})
print(async_pager.page_size)
print(async_pager[0])
await async_pager.next_page()
print(async_pager[0])
```

--------------------------------

TITLE: List Tuned Generative AI Models Asynchronously with Python
DESCRIPTION: These snippets show how to asynchronously retrieve a list of tuned Generative AI models. The first uses `async for` for direct iteration, and the second demonstrates asynchronous paging to manage large result sets.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_70

LANGUAGE: python
CODE:
```
async for job in await client.aio.models.list(config={'page_size': 10, 'query_base': False}):
    print(job)
```

LANGUAGE: python
CODE:
```
async_pager = await client.aio.models.list(config={'page_size': 10, 'query_base': False})
print(async_pager.page_size)
print(async_pager[0])
await async_pager.next_page()
print(async_pager[0])
```

--------------------------------

TITLE: Tune a Generative AI Model with Python
DESCRIPTION: This snippet demonstrates how to initiate a tuning job for a Generative AI model using the Google GenAI Python client. It specifies a base model, a training dataset from a GCS URI, and configuration for the tuning process, including the number of epochs and a display name for the tuned model.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_63

LANGUAGE: python
CODE:
```
gcs_uri='gs://cloud-samples-data/ai-platform/generative_ai/gemini-1_5/text/sft_train_data.jsonl'

from google.genai import types

tuning_job = client.tunings.tune(
    base_model=model,
    training_dataset=training_dataset,
    config=types.CreateTuningJobConfig(
        epoch_count=1, tuned_model_display_name='test_dataset_examples model'
    ),
)
```

--------------------------------

TITLE: Generate Image from Text Prompt (Python)
DESCRIPTION: Shows how to generate an image from a text prompt using `client.models.generate_images`. It includes configuration options like the number of images, inclusion of RAI reason, and desired output MIME type, then displays the first generated image.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_47

LANGUAGE: python
CODE:
```
from google.genai import types

# Generate Image
response1 = client.models.generate_images(
    model='imagen-3.0-generate-002',
    prompt='An umbrella in the foreground, and a rainy night sky in the background',
    config=types.GenerateImagesConfig(
        number_of_images=1,
        include_rai_reason=True,
        output_mime_type='image/jpeg',
    ),
)
response1.generated_images[0].image.show()
```

--------------------------------

TITLE: List Generative AI Tuning Jobs Asynchronously with Python
DESCRIPTION: These snippets demonstrate how to asynchronously retrieve a list of tuning jobs. They show both direct asynchronous iteration and the use of an asynchronous pager for handling large result sets efficiently.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_73

LANGUAGE: python
CODE:
```
async for job in await client.aio.tunings.list(config={'page_size': 10}):
    print(job)
```

LANGUAGE: python
CODE:
```
async_pager = await client.aio.tunings.list(config={'page_size': 10})
print(async_pager.page_size)
print(async_pager[0])
await async_pager.next_page()
print(async_pager[0])
```

--------------------------------

TITLE: Configure Socks5 Proxy for Google Gen AI Client
DESCRIPTION: This Python code demonstrates how to configure a SOCKS5 proxy for the Google Gen AI client by passing proxy details directly to `client_args` and `async_client_args` within `http_options`, enabling secure and flexible network routing for both sync and async operations.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_11

LANGUAGE: python
CODE:
```
http_options = types.HttpOptions(
    client_args={'proxy': 'socks5://user:pass@host:port'},
    async_client_args={'proxy': 'socks5://user:pass@host:port'},
)

client=Client(..., http_options=http_options)
```

--------------------------------

TITLE: Generate content from text using Google Generative AI Python SDK
DESCRIPTION: Demonstrates how to generate text content from a given prompt using the `client.models.generate_content` method. It takes a model name and a text prompt as input and prints the generated response.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_12

LANGUAGE: python
CODE:
```
response = client.models.generate_content(
    model='gemini-2.0-flash-001', contents='Why is the sky blue?'
)
print(response.text)
```

--------------------------------

TITLE: Generate Images with Imagen Models using Google GenAI Python Client
DESCRIPTION: This snippet demonstrates how to generate images using Google's Imagen models via the GenAI Python client. It covers specifying the model, providing a text prompt, and configuring image generation parameters such as the number of images, output format, and aspect ratio.

SOURCE: https://github.com/googleapis/python-genai/blob/main/codegen_instructions.md#_snippet_19

LANGUAGE: python
CODE:
```
from google import genai
from PIL import Image
from io import BytesIO

client = genai.Client()

result = client.models.generate_images(
    model='imagen-4.0-fast-generate-001',
    prompt="Image of a cat",
    config=dict(
        number_of_images=1, # 1 to 4 (always 1 for the ultra model)
        output_mime_type="image/jpeg",
        person_generation="ALLOW_ADULT" # 'ALLOW_ALL' (but not in Europe/Mena), 'DONT_ALLOW' or 'ALLOW_ADULT'
        aspect_ratio="1:1" # "1:1", "3:4", "4:3", "9:16", or "16:9"
    )
)

for generated_image in result.generated_images:
   image = Image.open(BytesIO(generated_image.image.image_bytes))
```

--------------------------------

TITLE: Generate content from an uploaded file using Google Generative AI Python SDK
DESCRIPTION: Illustrates how to upload a local file (`a11.txt`) using `client.files.upload` and then use the uploaded file's reference as part of the `contents` argument to generate a summary with `client.models.generate_content`.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_14

LANGUAGE: python
CODE:
```
file = client.files.upload(file='a11.txt')
response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents=['Could you summarize this file?', file]
)
print(response.text)
```

--------------------------------

TITLE: Stream Content Generation with Image Input (Python)
DESCRIPTION: Demonstrates how to stream content generation from the Gemini API, including an image as part of the input. It reads an image from a specified path, converts it to bytes, and sends it along with a text prompt to the `generate_content_stream` method, printing chunks as they arrive.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_39

LANGUAGE: python
CODE:
```
YOUR_IMAGE_MIME_TYPE = 'your_image_mime_type'
with open(YOUR_IMAGE_PATH, 'rb') as f:
    image_bytes = f.read()

for chunk in client.models.generate_content_stream(
    model='gemini-2.0-flash-001',
    contents=[
        'What is this image about?',
        types.Part.from_bytes(data=image_bytes, mime_type=YOUR_IMAGE_MIME_TYPE),
    ],
):
    print(chunk.text, end='')
```

--------------------------------

TITLE: Configure Environment Variables for Vertex AI API
DESCRIPTION: These bash commands set environment variables (`GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`) to configure the Google Gen AI client for use with the Vertex AI API, enabling automatic project and location detection.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_5

LANGUAGE: bash
CODE:
```
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT='your-project-id'
export GOOGLE_CLOUD_LOCATION='us-central1'
```

--------------------------------

TITLE: Create a Supervised Fine-Tuning Job (Python)
DESCRIPTION: Initiates a supervised fine-tuning job for a base model using a specified training dataset from GCS. It configures the tuning process, including the number of epochs and a display name for the resulting tuned model.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_54

LANGUAGE: python
CODE:
```
from google.genai import types

model = 'gemini-2.0-flash-001'
training_dataset = types.TuningDataset(
    gcs_uri='gs://cloud-samples-data/ai-platform/generative_ai/gemini-1_5/text/sft_train_data.jsonl',
)

from google.genai import types

tuning_job = client.tunings.tune(
    base_model=model,
    training_dataset=training_dataset,
    config=types.CreateTuningJobConfig(
        epoch_count=1, tuned_model_display_name='test_dataset_examples model'
    ),
)
print(tuning_job)
```

--------------------------------

TITLE: List tuned models asynchronously from Gemini API with Python
DESCRIPTION: This asynchronous Python snippet demonstrates how to list tuned models using the async API. It iterates through the results using `async for` and `await client.aio.models.list`, providing a non-blocking way to retrieve model information.

SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_20

LANGUAGE: python
CODE:
```
async for job in await client.aio.models.list(config={'page_size': 10, 'query_base': False}):
    print(job)
```

--------------------------------

TITLE: Configure GOOGLE_API_KEY Environment Variable for Gemini Developer API
DESCRIPTION: This bash command sets the `GOOGLE_API_KEY` environment variable, which allows the Google Gen AI client to automatically pick up the API key for authentication when using the Gemini Developer API without explicitly passing it in code.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_4

LANGUAGE: bash
CODE:
```
export GOOGLE_API_KEY='your-api-key'
```

--------------------------------

TITLE: Configure Proxy Environment Variables for Google Gen AI
DESCRIPTION: These bash commands set the `HTTPS_PROXY` and `SSL_CERT_FILE` environment variables, allowing the Google Gen AI SDK (using httpx or aiohttp) to route network requests through a specified HTTP proxy and use a custom SSL certificate for secure connections.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_10

LANGUAGE: bash
CODE:
```
export HTTPS_PROXY='http://username:password@proxy_uri:port'
export SSL_CERT_FILE='client.pem'
```

--------------------------------

TITLE: Generate Image with Google Gemini API (Python)
DESCRIPTION: This snippet demonstrates how to generate a new image using the `client.models.generate_images` method. It specifies the `imagen-3.0-generate-002` model, provides a textual prompt, and configures the output with parameters like the number of images, RAI reason inclusion, and output MIME type. Note that image generation support is currently behind an allowlist.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_39

LANGUAGE: python
CODE:
```
from google.genai import types

# Generate Image
response1 = client.models.generate_images(
    model='imagen-3.0-generate-002',
    prompt='An umbrella in the foreground, and a rainy night sky in the background',
    config=types.GenerateImagesConfig(
        number_of_images=1,
        include_rai_reason=True,
        output_mime_type='image/jpeg',
    ),
)
response1.generated_images[0].image.show()
```

--------------------------------

TITLE: Configure Content Generation (Python)
DESCRIPTION: Details the correct class for generation configuration, `types.GenerateContentConfig()`, and how `safety_settings` should be used within it, as opposed to the deprecated `genai.GenerationConfig()`.

SOURCE: https://github.com/googleapis/python-genai/blob/main/codegen_instructions.md#_snippet_4

LANGUAGE: Python
CODE:
```
genai.GenerationConfig(...)
```

LANGUAGE: Python
CODE:
```
types.GenerateContentConfig(...)
```

LANGUAGE: Python
CODE:
```
safety_settings={...}
```

--------------------------------

TITLE: Generate Content with a Tuned Model (Python)
DESCRIPTION: Shows how to use a fine-tuned model for content generation. It uses the `endpoint` of the completed tuning job's tuned model to make a `generate_content` call.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_56

LANGUAGE: python
CODE:
```
response = client.models.generate_content(
    model=tuning_job.tuned_model.endpoint,
    contents='why is the sky blue?',
)

print(response.text)
```

--------------------------------

TITLE: Configure Function Calling in Gemini API Request (Python)
DESCRIPTION: Illustrates a configuration snippet for the Google Gemini API's `generate_content` method, demonstrating how to enable automatic function calling, specify tools, set maximum remote calls, and define function calling mode within `GenerateContentConfig`.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_32

LANGUAGE: python
CODE:
```
        contents="What is the weather like in Boston?",
        config=types.GenerateContentConfig(
            tools=[get_current_weather],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=2
            ),
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode='ANY')
            ),
        ),
    )
```

--------------------------------

TITLE: Structure GenAI Content Argument with types.Content Instance in Python
DESCRIPTION: Demonstrates how to explicitly create a `types.Content` instance with a 'user' role and text parts using `types.Part.from_text`, which is the canonical way to structure the `contents` argument for `generate_content`.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_12

LANGUAGE: Python
CODE:
```
from google.genai import types

contents = types.Content(
    role='user',
    parts=[types.Part.from_text(text='Why is the sky blue?')]
)
```

--------------------------------

TITLE: Define training dataset for Gemini API model tuning with Python
DESCRIPTION: This Python snippet prepares a `TuningDataset` object for model fine-tuning. It specifies the dataset's source as a Google Cloud Storage (GCS) URI, which can point to either a JSONL file or a Vertex Multimodal Dataset for training.

SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_12

LANGUAGE: python
CODE:
```
from google.genai import types

model = 'gemini-2.0-flash-001'
training_dataset = types.TuningDataset(
  # or gcs_uri=my_vertex_multimodal_dataset
    gcs_uri='gs://cloud-samples-data/ai-platform/generative_ai/gemini-1_5/text/sft_train_data.jsonl',
)
```

--------------------------------

TITLE: Declare a Function for Google GenAI Tool Use (Python)
DESCRIPTION: This snippet demonstrates how to define a function's schema using `types.FunctionDeclaration` with a JSON schema for parameters and package it as a `types.Tool`. This tool is then passed to `client.models.generate_content` to enable function calling, and the model's response will contain the function call part.

SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_0

LANGUAGE: python
CODE:
```
from google.genai import types

function = types.FunctionDeclaration(
    name='get_current_weather',
    description='Get the current weather in a given location',
    parameters_json_schema={
        'type': 'object',
        'properties': {
            'location': {
                'type': 'string',
                'description': 'The city and state, e.g. San Francisco, CA',
            }
        },
        'required': ['location'],
    },
)

tool = types.Tool(function_declarations=[function])

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='What is the weather like in Boston?',
    config=types.GenerateContentConfig(tools=[tool]),
)

print(response.function_calls[0])
```

--------------------------------

TITLE: Generate Content from Uploaded File with GenAI Python SDK
DESCRIPTION: Illustrates how to upload a local file using `client.files.upload` and then use the uploaded file as part of the `contents` argument for `client.models.generate_content` with the Gemini Developer API.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_11

LANGUAGE: Python
CODE:
```
file = client.files.upload(file='a11.txt')
response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents=['Could you summarize this file?', file]
)
print(response.text)
```

--------------------------------

TITLE: Create Cached Content with Python GenAI API
DESCRIPTION: This code illustrates how to create cached content for a model using `client.caches.create`. It configures the cache with input files (either GCS URIs or previously uploaded files), a system instruction, a display name, and a time-to-live (TTL). Caching can improve performance for frequently used prompts.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_51

LANGUAGE: python
CODE:
```
from google.genai import types

if client.vertexai:
    file_uris = [
        'gs://cloud-samples-data/generative-ai/pdf/2312.11805v3.pdf',
        'gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf',
    ]
else:
    file_uris = [file1.uri, file2.uri]

cached_content = client.caches.create(
    model='gemini-2.0-flash-001',
    config=types.CreateCachedContentConfig(
        contents=[
            types.Content(
                role='user',
                parts=[
                    types.Part.from_uri(
                        file_uri=file_uris[0], mime_type='application/pdf'
                    ),
                    types.Part.from_uri(
                        file_uri=file_uris[1],
                        mime_type='application/pdf',
                    ),
                ],
            )
        ],
        system_instruction='What is the sum of the two pdfs?',
        display_name='test cache',
        ttl='3600s',
    ),
)
```

--------------------------------

TITLE: Structure `contents` with a list of function call parts for Google Generative AI Python SDK
DESCRIPTION: Shows how to provide multiple function calls as a list of `types.Part.from_function_call` instances. The SDK converts this list into a single `types.ModelContent` instance, allowing for multiple tool calls within one content block.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_19

LANGUAGE: python
CODE:
```
from google.genai import types

contents = [
    types.Part.from_function_call(
        name='get_weather_by_location',
        args={'location': 'Boston'}
    ),
    types.Part.from_function_call(
        name='get_weather_by_location',
        args={'location': 'New York'}
    ),
]
```

--------------------------------

TITLE: List Batch Jobs with Pager (Synchronous, Python)
DESCRIPTION: This snippet illustrates using a pager object for synchronous listing of batch jobs. It fetches a page of results, accesses elements, and then navigates to the next page using `pager.next_page()`. This provides more control over pagination than simple iteration.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_82

LANGUAGE: python
CODE:
```
from google.genai import types

    pager = client.batches.list(config=types.ListBatchJobsConfig(page_size=10))
    print(pager.page_size)
    print(pager[0])
    pager.next_page()
    print(pager[0])
```

--------------------------------

TITLE: List Tuning Jobs Asynchronously using Python
DESCRIPTION: This snippet demonstrates how to list tuning jobs asynchronously using `client.aio.tunings.list`. It shows both iterating through the results with `async for` and using an asynchronous pager object to access results and navigate pages.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_6

LANGUAGE: python
CODE:
```
async for job in await client.aio.tunings.list(config={'page_size': 10}):
    print(job)

async_pager = await client.aio.tunings.list(config={'page_size': 10})
print(async_pager.page_size)
print(async_pager[0])
await async_pager.next_page()
print(async_pager[0])
```

--------------------------------

TITLE: Embed Multiple Text Contents with Configuration (Python)
DESCRIPTION: Demonstrates how to generate embeddings for multiple text contents simultaneously using `client.models.embed_content`. It also shows how to include an optional configuration, such as specifying the `output_dimensionality` for the embeddings.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_46

LANGUAGE: python
CODE:
```
from google.genai import types

# multiple contents with config
response = client.models.embed_content(
    model='text-embedding-004',
    contents=['why is the sky blue?', 'What is your age?'],
    config=types.EmbedContentConfig(output_dimensionality=10),
)

print(response)
```

--------------------------------

TITLE: List Batch Prediction Jobs Asynchronously with Python
DESCRIPTION: Illustrates how to asynchronously list batch prediction jobs using `client.aio.batches.list`. This method allows for efficient iteration over job results.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_67

LANGUAGE: python
CODE:
```
from google.genai import types

async for job in await client.aio.batches.list(
    config=types.ListBatchJobsConfig(page_size=10)
):
    print(job)
```

--------------------------------

TITLE: Structure `contents` with `types.Content` for Google Generative AI Python SDK
DESCRIPTION: Demonstrates the canonical way to provide content to `generate_content` by explicitly creating a `types.Content` instance with a specified role and parts. This method offers fine-grained control over the input structure.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_15

LANGUAGE: python
CODE:
```
from google.genai import types

contents = types.Content(
    role='user',
    parts=[types.Part.from_text(text='Why is the sky blue?')]
)
```

--------------------------------

TITLE: Create cached content for Gemini API with Python
DESCRIPTION: This Python snippet shows how to create cached content using `client.caches.create`. It configures the cache with a model, content parts derived from file URIs (distinguishing between Vertex AI and non-Vertex AI sources), a system instruction, display name, and a time-to-live (TTL).

SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_9

LANGUAGE: python
CODE:
```
from google.genai import types

if client.vertexai:
    file_uris = [
        'gs://cloud-samples-data/generative-ai/pdf/2312.11805v3.pdf',
        'gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf',
    ]
else:
    file_uris = [file1.uri, file2.uri]

cached_content = client.caches.create(
    model='gemini-2.0-flash-001',
    config=types.CreateCachedContentConfig(
        contents=[
            types.Content(
                role='user',
                parts=[
                    types.Part.from_uri(
                        file_uri=file_uris[0], mime_type='application/pdf'
                    ),
                    types.Part.from_uri(
                        file_uri=file_uris[1],
                        mime_type='application/pdf',
                    ),
                ],
            )
        ],
        system_instruction='What is the sum of the two pdfs?',
        display_name='test cache',
        ttl='3600s',
    ),
)
```

--------------------------------

TITLE: Handle API Errors (Python)
DESCRIPTION: Provides the correct import path for handling API errors from the Google GenAI SDK, `google.genai.errors.APIError`, replacing the older `google.api_core.exceptions.GoogleAPIError`.

SOURCE: https://github.com/googleapis/python-genai/blob/main/codegen_instructions.md#_snippet_5

LANGUAGE: Python
CODE:
```
from google.api_core.exceptions import GoogleAPIError
```

LANGUAGE: Python
CODE:
```
from google.genai.errors import APIError
```

--------------------------------

TITLE: Upload File and Create Batch Prediction Job with Python
DESCRIPTION: This snippet demonstrates the process of uploading a JSONL file containing batch prediction requests to the Gemini Developer API, and then creating a batch job using the uploaded file's name as the source.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_77

LANGUAGE: python
CODE:
```
# Upload a file to Gemini Developer API
file_name = client.files.upload(
    file='myrequest.json',
    config=types.UploadFileConfig(display_name='test_json'),
)
# Create a batch job with file name
```

--------------------------------

TITLE: Structure `contents` with a string for Google Generative AI Python SDK
DESCRIPTION: Shows how a simple string input for the `contents` argument is automatically converted by the SDK into a `types.UserContent` instance with a single text part. This is a convenient shorthand for basic text prompts.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_16

LANGUAGE: python
CODE:
```
contents='Why is the sky blue?'
```

--------------------------------

TITLE: Create Cached Content for Generative AI (Python)
DESCRIPTION: This Python code demonstrates how to create cached content for a generative AI model. It configures the cache with input content (e.g., PDFs), a system instruction, a display name, and a time-to-live (TTL) for efficient reuse.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_59

LANGUAGE: python
CODE:
```
from google.genai import types

if client.vertexai:
    file_uris = [
        'gs://cloud-samples-data/generative-ai/pdf/2312.11805v3.pdf',
        'gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf',
    ]
else:
    file_uris = [file1.uri, file2.uri]

cached_content = client.caches.create(
    model='gemini-2.0-flash-001',
    config=types.CreateCachedContentConfig(
        contents=[
            types.Content(
                role='user',
                parts=[
                    types.Part.from_uri(
                        file_uri=file_uris[0], mime_type='application/pdf'
                    ),
                    types.Part.from_uri(
                        file_uri=file_uris[1],
                        mime_type='application/pdf',
                    ),
                ],
            )
        ],
        system_instruction='What is the sum of the two pdfs?',
        display_name='test cache',
        ttl='3600s',
    ),
)
```

--------------------------------

TITLE: Edit Image with Masking (Python)
DESCRIPTION: Illustrates how to edit an image using `client.models.edit_image`, including the use of reference images and masking. It demonstrates creating a background mask and applying an inpainting insertion edit based on a new prompt. This functionality is supported in Vertex AI.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_49

LANGUAGE: python
CODE:
```
# Edit the generated image from above
from google.genai import types
from google.genai.types import RawReferenceImage, MaskReferenceImage

raw_ref_image = RawReferenceImage(
    reference_id=1,
    reference_image=response1.generated_images[0].image,
)

# Model computes a mask of the background
mask_ref_image = MaskReferenceImage(
    reference_id=2,
    config=types.MaskReferenceConfig(
        mask_mode='MASK_MODE_BACKGROUND',
        mask_dilation=0,
    ),
)

response3 = client.models.edit_image(
    model='imagen-3.0-capability-001',
    prompt='Sunlight and clear sky',
    reference_images=[raw_ref_image, mask_ref_image],
    config=types.EditImageConfig(
        edit_mode='EDIT_MODE_INPAINT_INSERTION',
        number_of_images=1,
        include_rai_reason=True,
        output_mime_type='image/jpeg',
    ),
)
response3.generated_images[0].image.show()
```

--------------------------------

TITLE: Define User Content with Mixed Parts (Python)
DESCRIPTION: Shows how to construct `types.UserContent` with a list of `types.Part` objects, combining text and image URI parts for multimodal input to the Generative AI model. This structure is essential for sending diverse content types to the model.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_22

LANGUAGE: python
CODE:
```
[
    types.UserContent(
        parts=[
        types.Part.from_text('What is this image about?'),
        types.Part.from_uri(
            file_uri: 'gs://generativeai-downloads/images/scones.jpg',
            mime_type: 'image/jpeg',
        )
        ]
    )
]
```

--------------------------------

TITLE: Upload files to Google Gemini API with Python
DESCRIPTION: This Python snippet shows how to upload local files to the Gemini Developer API using the client's `files.upload` method. It takes a local file path as input and returns file objects, which are then printed to confirm successful upload.

SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_6

LANGUAGE: python
CODE:
```
file1 = client.files.upload(file='2312.11805v3.pdf')
file2 = client.files.upload(file='2403.05530.pdf')

print(file1)
print(file2)
```

--------------------------------

TITLE: List Tuned Models Asynchronously using Python
DESCRIPTION: This snippet demonstrates how to list tuned models asynchronously using `client.aio.models.list`. It shows both iterating through the results with `async for` and using an asynchronous pager object to access results and navigate pages.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/index.html#_snippet_3

LANGUAGE: python
CODE:
```
async for job in await client.aio.models.list(config={'page_size': 10, 'query_base': False}):
    print(job)

async_pager = await client.aio.models.list(config={'page_size': 10, 'query_base': False})
print(async_pager.page_size)
print(async_pager[0])
await async_pager.next_page()
print(async_pager[0])
```

--------------------------------

TITLE: Stream Text Content from Gemini API in Python
DESCRIPTION: Illustrates how to perform synchronous streaming for text generation using the Google Gemini API's `generate_content_stream` method, allowing for real-time processing of model outputs.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_37

LANGUAGE: python
CODE:
```
for chunk in client.models.generate_content_stream(
    model='gemini-2.0-flash-001', contents='Tell me a story in 300 words.'
):
    print(chunk.text, end='')
```

--------------------------------

TITLE: List tuned models from Gemini API with Python (Iterator)
DESCRIPTION: This Python snippet iterates through and prints a list of tuned models available in the Gemini API. It uses `client.models.list` with a configuration to specify page size and exclude base models, providing an iterable for processing each model.

SOURCE: https://github.com/googleapis/python-genai/blob/main/README.md#_snippet_18

LANGUAGE: python
CODE:
```
for model in client.models.list(config={'page_size': 10, 'query_base': False}):
    print(model)
```

--------------------------------

TITLE: Generate Text Content Asynchronously (Python)
DESCRIPTION: Shows how to perform an asynchronous, non-streaming text generation request using the `client.aio.models.generate_content` method. It sends a text prompt to the specified model and prints the complete response text once available.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_40

LANGUAGE: python
CODE:
```
response = await client.aio.models.generate_content(
    model='gemini-2.0-flash-001', contents='Tell me a story in 300 words.'
)

print(response.text)
```

--------------------------------

TITLE: Apply Safety Settings to Generative AI Content Generation (Python)
DESCRIPTION: Illustrates how to configure safety settings for content generation, allowing users to specify categories (e.g., hate speech) and thresholds (e.g., `BLOCK_ONLY_HIGH`) for blocking harmful content. This helps ensure responsible AI usage.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_27

LANGUAGE: python
CODE:
```
from google.genai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='Say something bad.',
    config=types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(
                category='HARM_CATEGORY_HATE_SPEECH',
                threshold='BLOCK_ONLY_HIGH',
            )
        ]
    ),
)
print(response.text)
```

--------------------------------

TITLE: Generate content with various data types using Google GenAI Python
DESCRIPTION: This snippet illustrates how to use `Part.from_bytes` to pass a variety of data types, such as images, audio, video, or PDF files, as byte streams to the model. It reads an image file into bytes and includes it in the `contents` list with its MIME type.

SOURCE: https://github.com/googleapis/python-genai/blob/main/codegen_instructions.md#_snippet_9

LANGUAGE: python
CODE:
```
from google.genai import types

  with open('path/to/small-sample.jpg', 'rb') as f:
    image_bytes = f.read()

  response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[
      types.Part.from_bytes(
        data=image_bytes,
        mime_type='image/jpeg',
      ),
      'Caption this image.'
    ]
  )

  print(response.text)
```

--------------------------------

TITLE: Upload Files to Google Generative AI (Python)
DESCRIPTION: This Python code demonstrates uploading local files to the Google Generative AI service. It uses the `client.files.upload` method to make the files available for API operations, such as using them in content generation or caching.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_56

LANGUAGE: python
CODE:
```
file1 = client.files.upload(file='2312.11805v3.pdf')
file2 = client.files.upload(file='2403.05530.pdf')

print(file1)
print(file2)
```

--------------------------------

TITLE: Stream Image Content from Google Cloud Storage with Gemini API in Python
DESCRIPTION: Demonstrates how to stream content from the Google Gemini API when the input includes an image stored in Google Cloud Storage, using `types.Part.from_uri` to reference the image.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_38

LANGUAGE: python
CODE:
```
from google.genai import types

for chunk in client.models.generate_content_stream(
    model='gemini-2.0-flash-001',
    contents=[
        'What is this image about?',
        types.Part.from_uri(
            file_uri='gs://generativeai-downloads/images/scones.jpg',
            mime_type='image/jpeg',
        ),
    ],
):
    print(chunk.text, end='')
```

--------------------------------

TITLE: List Batch Jobs with Pager (Asynchronous, Python)
DESCRIPTION: This snippet demonstrates asynchronous pagination for listing batch jobs. It obtains an asynchronous pager object, accesses elements, and then asynchronously fetches the next page using `await async_pager.next_page()`. This is ideal for integrating with asynchronous workflows.

SOURCE: https://github.com/googleapis/python-genai/blob/main/docs/_sources/index.rst.txt#_snippet_84

LANGUAGE: python
CODE:
```
from google.genai import types

    async_pager = await client.aio.batches.list(
        config=types.ListBatchJobsConfig(page_size=10)
    )
    print(async_pager.page_size)
    print(async_pager[0])
    await async_pager.next_page()
    print(async_pager[0])
```