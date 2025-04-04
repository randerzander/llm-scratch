from discord.ext import commands
import discord
import asyncio

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)


@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')
    await download_channel_history()

async def download_channel_history():
    for channel_id in [
        982073726826840066, #stuff
        982073988350103572, #homelab
        982074048647426108, #food
        982074004619821117, #gaming
        982074818084089858, #dev
        1010560173359759410, #pics
        982075959178051665, #pets
        1017967859437146152, #music
        1079293058069299250, #prompt-jockeying
        1210667544973680660, #ml
        1210719039198855258, #movie-club
        1216021628010369034, #vj-random
        1308637302658175006, #bot-stuff
        1315022823361876029, #thunderdome
    ]:
        channel = bot.get_channel(channel_id)
        print(f"Downloading channel history for {channel.name}")
        
        if not channel:
            print("Channel not found.")
            return

        messages = []
        async for message in channel.history(limit=None):
            display_name = message.author.display_name  # Global display name
            if hasattr(message, "member") and message.member is not None:
                display_name = message.member.display_name
            messages.append(f"{message.created_at}|_|{message.author}|{display_name}|_|{message.content}_|_")

        with open(f"channel_logs/{channel.name}_{channel_id}.txt", "w", encoding="utf-8") as f:
            for msg in reversed(messages):
                f.write(f"{msg}\n")

        print(f"Channel history saved to {channel.name}_{channel_id}.txt")

bot.run(open("bot_token").read().strip())
