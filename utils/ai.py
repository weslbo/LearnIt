from __future__ import print_function
from IPython.core.magic import (Magics, magics_class, line_magic, cell_magic, line_cell_magic)
from IPython import get_ipython
from IPython.display import Audio, display
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatHistory
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings, OpenAIChatPromptExecutionSettings
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.prompt_template.input_variable import InputVariable
from semantic_kernel.functions import KernelArguments
from IPython.display import display, Markdown, Latex
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os, fnmatch
import requests
import markdownify
import re
import azure.cognitiveservices.speech as speechsdk
import datetime
import json
import uuid
import time

kernel = Kernel()

kernel.add_service(
    AzureChatCompletion(
        service_id="default",
    ),
)

prompt = """
Assistant can have a conversation with you about any topic.
It can give explicit instructions or say 'I don't know' if it does not have an answer.

{{$history}}
User: {{$user_input}}
Assistant: """

execution_settings = AzureChatPromptExecutionSettings(
        service_id="default",
        ai_model_id="gpt-4o",
        max_tokens=2000,
        temperature=0.7,
    )

prompt_template_config = PromptTemplateConfig(
    template=prompt,
    name="chat",
    template_format="semantic-kernel",
    input_variables=[
        InputVariable(name="user_input", description="The user input", is_required=True),
        InputVariable(name="history", description="The conversation history", is_required=True),
    ],
    execution_settings=execution_settings,
)

chat_function = kernel.add_function(
    function_name="chat",
    plugin_name="chatPlugin",
    prompt_template_config=prompt_template_config,
)

chat_history = ChatHistory()
chat_history.add_system_message("You are a helpful AI Assistant. Answer to the point and limit your output so your answers are simple to understand. Highlight the most important keywords in **bold**.")


# The class MUST call this class decorator at creation time
@magics_class
class MyMagics(Magics):

    async def questionasync(self, cell):
        #global history

        # Process the user message and get an answer
        answer = await kernel.invoke(chat_function, KernelArguments(user_input=cell, history=chat_history))

        # Show the response
        display(Markdown(str(answer)))

        chat_history.add_user_message(cell)
        chat_history.add_assistant_message(str(answer))

    async def audioasync(self, cell):
        #global history

        msg = "Answer the following question using natural human spoken language. Don't use any bullet points, code fragments, markdown etc, just natural spoken language: " + cell

        # Process the user message and get an answer
        answer = await kernel.invoke(chat_function, KernelArguments(user_input=msg, history=chat_history))

        strssml = f"""
            <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
                <voice name="en-US-AndrewMultilingualNeural">
                    {str(cell)}
                </voice>
                <voice name="en-US-AvaMultilingualNeural">
                    {str(answer)}
                </voice>
            </speak>
        """

        service_region = os.getenv("SPEECH_REGION")
        speech_key = os.getenv("SPEECH_API_KEY")
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
        speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Audio24Khz96KBitRateMonoMp3)  

        mp3_filename = f"./audio/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}.mp4"
        file_config = speechsdk.audio.AudioOutputConfig(filename=mp3_filename)
        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=file_config)  
        result = speech_synthesizer.speak_ssml_async(strssml).get()
    
        # Show the response
        display(Markdown(str(answer)))
        # display(Audio(mp3_filename, autoplay=True))
        display(Video(mp3_filename, autoplay=True))

        chat_history.add_user_message(cell)
        chat_history.add_assistant_message(str(answer))

    async def mindmapasync(self, cell):
        #global history

        msg = cell + " . Create a mindmap using plantuml. Use the following format: ```plantuml @startmindmap <the mindmap> @endmindmap ```. Only return the mindmap "

        # Process the user message and get an answer
        answer = await kernel.invoke(chat_function, KernelArguments(user_input=msg, history=chat_history))

        # Show the response
        display(Markdown(str(answer)))

        chat_history.add_user_message(cell)
        chat_history.add_assistant_message(str(answer))
        
    @cell_magic
    def question(self, line, cell):
        """
        Custom magic command for interacting with Azure OpenAI GPT model.
        Keeps track of conversation history.
        """
        # Wrap the coroutine call using asyncio.run or an event loop
        import nest_asyncio
        import asyncio
        nest_asyncio.apply()
        return asyncio.run(self.questionasync(cell))
    
    @cell_magic
    def audio(self, line, cell):
        """
        Custom magic command for interacting with Azure OpenAI GPT model.
        Keeps track of conversation history.
        """
        # Wrap the coroutine call using asyncio.run or an event loop
        import nest_asyncio
        import asyncio
        nest_asyncio.apply()
        return asyncio.run(self.audioasync(cell))
    
    @cell_magic
    def video(self, line, cell):
        """
        Custom magic command for interacting with Azure OpenAI GPT model.
        Keeps track of conversation history.
        """

        # Wrap the coroutine call using asyncio.run or an event loop
        import nest_asyncio
        import asyncio
        nest_asyncio.apply()
        return asyncio.run(self.videoasync(cell))
    
  
    async def videoasync(self, cell):
        """
        Custom magic command for interacting with Azure OpenAI GPT model.
        Keeps track of conversation history.
        """

        # Process the user message and get an answer
        answer = await kernel.invoke(chat_function, KernelArguments(user_input=cell, history=chat_history))

        # Show the response
        display(Markdown(str(answer)))

        chat_history.add_user_message(cell)
        chat_history.add_assistant_message(str(answer))

        # Wrap the coroutine call using asyncio.run or an event loop
        job_id = str(uuid.uuid4())

        url = f'https://{os.getenv("SPEECH_REGION")}.api.cognitive.microsoft.com/avatar/batchsyntheses/{job_id}?api-version=2024-04-15-preview'
        print(url)
        header = {
            'Content-Type': 'application/json',
            'Ocp-Apim-Subscription-Key': os.getenv("SPEECH_API_KEY")
        }

        payload = {
            'synthesisConfig': {
                "voice": f'en-US-EmmaNeural',
            },
            'customVoices': {
                # "YOUR_CUSTOM_VOICE_NAME": "YOUR_CUSTOM_VOICE_ID"
            },
            "inputKind": "plainText",
            "inputs": [
                {
                    "content": answer,
                },
            ],
            "avatarConfig":
                {
                    "customized": False, # set to True if you want to use customized avatar
                    "talkingAvatarCharacter": 'Lori',  # talking avatar character
                    "talkingAvatarStyle": 'graceful',  # talking avatar style, required for prebuilt avatar, optional for custom avatar
                    "videoFormat": "webm",  # mp4 or webm, webm is required for transparent background // was mp4
                    "videoCodec": "vp9",  # hevc, h264 or vp9, vp9 is required for transparent background; default is hevc // was h264
                    "subtitleType": "soft_embedded",
                    "backgroundColor": "transparent", # background color in RGBA format, default is white; can be set to 'transparent' for transparent background
                    "videoCrop": {  "topLeft": { "x": 460, "y": 0}, "bottomRight": { "x": 1460, "y": 1079}  }
                }  
        }
        
        response = requests.put(url, json.dumps(payload, default=str), headers=header)
        if response.status_code < 400:
            print('Batch avatar synthesis job submitted successfully')
            print(f'Job ID: {response.json()["id"]}')
        else:
            print(f'Failed to submit batch avatar synthesis job: [{response.status_code}], {response.text}')

        while True:
            status = MyMagics.get_synthesis(url)
            if status == 'Succeeded':
                print('batch avatar synthesis job succeeded')
                break
            elif status == 'Failed':
                print('batch avatar synthesis job failed')
                break
            else:
                print(f'batch avatar synthesis job is [{status}]')
                time.sleep(5)        
                    
    
    @cell_magic
    def mindmap(self, line, cell):
        """
        Custom magic command for interacting with Azure OpenAI GPT model.
        Keeps track of conversation history.
        """
        # Wrap the coroutine call using asyncio.run or an event loop
        import nest_asyncio
        import asyncio
        nest_asyncio.apply()
        return asyncio.run(self.mindmapasync(cell))
    
    @cell_magic
    def learn(self, line, cell):
        """
        Custom magic command for learning content from Microsoft Learn.
        """
        learnmoduleurl = str(cell).replace('\n', '')

        learn_module = requests.get(learnmoduleurl)
        soup_learnmodule = BeautifulSoup(learn_module.text, "html.parser")
        links_units = soup_learnmodule.find_all(class_="unit-title")
        links_units = [link for link in links_units if not any(keyword in link["href"] for keyword in ["exercise", "knowledge-check", "summary"])]
        title_module = soup_learnmodule.find("h1", class_="title").text
        absolute_urls_units = [urljoin(cell, link["href"]) for link in links_units]

        allcontent = ""

        for msg in chat_history:
            chat_history.remove_message(msg)

        for url in absolute_urls_units:
            print(url)
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')

            # might need to adapt this when working with other web pages (not Microsoft Learn)
            div = soup.find(id="unit-inner-section")

            for ul in div.find_all("ul", class_="metadata"):
                ul.decompose()
            for d in div.find_all("div", class_="xp-tag"):
                d.decompose()
            for next in div.find_all("div", class_="next-section"):
                next.decompose()
            for header in div.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
                header.string = "\n# " + header.get_text() + "\n"
            for code in div.find_all("code"):
                code.decompose()

            markdown = markdownify.markdownify(str(div), heading_style="ATX", bullets="-")
            markdown = re.sub('\n{3,}', '\n\n', markdown)
            markdown = markdown.replace("[Continue](/en-us/)", "")

            allcontent += title_module + "\n\n" + markdown + "\n\n"
    
            chat_history.add_system_message("You are a helpful AI Assistant. Answer to the point and limit your output so your answers are simple to understand. Highlight the most important keywords in **bold**. Here's the content of the module: " + allcontent)

        return line, cell

      
    @classmethod
    def get_synthesis(cls, url):
        header = {
            'Ocp-Apim-Subscription-Key': os.getenv("SPEECH_API_KEY")
        }

        response = requests.get(url, headers=header)
        if response.status_code < 400:
            print('Get batch synthesis job successfully')
            print(response.json())
            if response.json()['status'] == 'Succeeded':
                print(f'Batch synthesis job succeeded, download URL: {response.json()["outputs"]["result"]}')
            return response.json()['status']
        else:
            print(f'Failed to get batch synthesis job: {response.text}')        

def load_ipython_extension(ipython):
    """
    Any module file that define a function named `load_ipython_extension`
    can be loaded via `%load_ext module.path` or be configured to be
    autoloaded by IPython at startup time.
    """
    # You can register the class itself without instantiating it.  IPython will
    # call the default constructor on it.
    ipython.register_magics(MyMagics)

ip = get_ipython()
load_ipython_extension(ip)
