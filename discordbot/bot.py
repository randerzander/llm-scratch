#from llm_scratch import llama_cpp as llm
from llm_scratch import gpt_4o as llm
from ragatouille import RAGPretrainedModel

import discord
import os, time

# Set up intents
intents = discord.Intents.default()
intents.message_content = True

# Create a client instance
client = discord.Client(intents=intents)

# Channel ID where you want to send the message
CHANNEL_ID = 1308637302658175006

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')
    # Get the channel
    #channel = client.get_channel(CHANNEL_ID)
    #if channel:
    #    await channel.send(llm("Write a 'Is this thing on?' type mic test for a chatbot"))
    #else:
    #@client.event

@client.event
async def on_message(message):
    # don't resond to ourselves
    if message.author == client.user:
        return

    if client.user.mentioned_in(message):
        msg = message.content.split(">")[1].strip()
        #cmd = "Is the the above message [helpful, hurtful, condescending, questioning, joking, argumentative]. Answer in as few words as possible"
        #classifier = gpt_4o(f"{msg}\n\n{cmd}")
        #relevant_poems = lookup()
        #resp = llm(f"Write a 3 stanza max no yapping poem in the style of tennyson to {txt}")
        #thread = await message.create_thread(name="response")
        #await thread.send(resp)

        cont = RAG.search(query=msg, k=10)
        rel_cont = "\n\n".join([c["content"] for c in cont])
        resp = llm(f"Write a 3 stanza max no yapping poem to address {msg}\n\n{rel_cont}")
        await message.channel.send(f"{message.author.mention}: {resp}")

def load_rag(index_path):
    return RAGPretrainedModel.from_index(index_path)

# Run the client
TOKEN = open("token.txt", "r").read()

txt = open("prompt_jockeying.txt").read()
messages = txt.split("_|_\n")[:-1]

index_path = '.ragatouille/colbert/indexes/combo_context'
RAG = load_rag(index_path)
#cont = RAG.search(query="@UseFool What does randerzaner think about colbert", k=10)
#rel_cont = "\n\n".join([c["content"] for c in cont])
#llm(f"Write a 3 stanza max no yapping poem in the style of tennyson to {rel_cont}")
client.run(TOKEN)
