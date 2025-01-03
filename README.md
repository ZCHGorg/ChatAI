ALPHA VERSION doesn't work great, but it should at least deploy well and help you get started.  WARNING: will flood swapfile as of latest version.  Proceed with caution in an isolated environment.  I recommend vultr.com Centos8 node

# 🌟 Unleash the Power of Self-Improving Bots! 🚀

Welcome to ZchgAIBot – a cutting-edge platform that brings the future to your fingertips. Prepare to be amazed as our Self-Improving Bot evolves before your eyes, delivering more intelligent and sophisticated interactions with each conversation.

## What is ZchgAIBot?

ZchgAIBot is more than just a bot; it's a window into the next era of AI interaction. We've harnessed the capabilities of OpenAI's state-of-the-art GPT-3 model and paired it with ingenious self-improvement mechanisms. The outcome? A bot that learns, adapts, and polishes its responses, making each interaction truly unique.

## Key Features

- **Unparalleled Learning:** Witness your bot learn from user feedback and its own conversations, honing its responses to perfection.
  
- **Seamless Integration:** Effortlessly deploy and run the Self-Improving Bot using Docker, ensuring a seamless experience across various systems.

- **Dynamic Context:** Our bot retains prior interactions, maintaining context to provide more pertinent and captivating responses.

- **Resource Optimization:** Intelligent resource management guarantees peak bot performance, regardless of the conversation's intricacy.

- **User Feedback Loop:** The bot refines its conversational prowess based on user feedback, progressively enhancing its abilities.

## Use Cases

ZchgAIBot's potential is boundless, presenting value across diverse domains:

- **Customer Support:** Revolutionize customer interactions with an AI assistant that offers intuitive solutions and learns from each support ticket.

- **Education:** Craft a dynamic learning companion that engages students in personalized, educational dialogues.

- **Creative Writing:** Collaborate with the bot to brainstorm ideas, receive suggestions, and elevate your creative writing endeavors.

- **Problem Solving:** Turn to ZchgAIBot for inventive problem-solving discussions, where the bot's insights could be the game-changer.

## Getting Started

To embark on this exciting journey, refer to our comprehensive installation guide below. Set up your Self-Improving Bot and witness AI evolution firsthand.

## Connect with Us

Stay updated and share your experiences on social media using **#ZchgAIBotAI**. We're eager to see the remarkable interactions you create!

---

## Self-Improving Bot Installation Guide

Welcome to the installation guide for the Self-Improving Bot! We're thrilled to have you on board. To get this innovative bot up and running, follow these steps:

### Prepare Your Environment

Designed for CentOS 8

First, ensure that you have all the required files in the same directory, including `deploy_bot.sh`, `requirements.txt`, `self_improving_bot.py`, and `Dockerfile`.

Remember to replace `'GIVE_ME_YOUR_API_KEY_HERE'` with your Chat GPT API key!

### Install Docker with Ease

We've simplified the process with a convenient shell script. Open your terminal, navigate to the directory containing the files, and execute the following commands:

```shell
chmod +x deploy_bot.sh
./deploy_bot.sh
```
This script will handle Docker installation, ensuring you're ready to go in no time.

Build and Launch the Bot
Once Docker is set up, it's time to build and launch your Self-Improving Bot. Run these commands sequentially:

```shell
docker build -t self_improving_bot -f Dockerfile --no-cache .
```

```shell
docker run -it --rm --network host --privileged --name self_improving_bot_container self_improving_bot
```

Your bot will spring to life, poised to engage in conversations and learn from interactions.

Ensure Compatibility-
If issues arise due to file conversions from Windows to Linux, don't fret – we've got you covered. Execute the following commands to address any concerns:

```shell
sudo yum install dos2unix
dos2unix deploy_bot.sh requirements.txt self_improving_bot.py Dockerfile
```

These commands guarantee a seamless experience by formatting everything perfectly.

Now you're all set to witness the magic of the Self-Improving Bot! Engage in conversations, observe its learning, and watch its responses evolve over time. Feel free to explore and customize the bot's behavior to match your unique use cases.

Should questions arise or issues arise, don't hesitate to reach out to our support team. Happy bot-building! 🤖🌟

<p align="center">
  Made with ❤️ by the innovators at https://forum.zchg.org
</p>

![image](https://github.com/ZCHGorg/ChatAI/assets/24325826/04302ee3-0617-4272-a072-8ce6267488a0)

