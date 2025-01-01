import discord
import os

from llm_scratch import llama_cpp as llm
from llm_scratch import gpt_4o

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
        cmd = "Is the the above message [helpful, hurtful, condescending, questioning, joking, argumentative]. Answer in as few words as possible"
        classifier = gpt_4o(f"{msg}\n\n{cmd}")

        relevant_poems = lookup()

        resp = llm(f"Write a 3 stanza max no yapping poem in the style of tennyson to {txt}")
        thread = await message.create_thread(name="response")
        await thread.send(resp)
        #await message.channel.send(f"{message.author.mention}: {resp}")

# Run the client
TOKEN = open("token.txt", "r").read()
client.run(TOKEN)
