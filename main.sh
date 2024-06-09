#!/bin/bash
python main.py &
python LLM/chatgpt.py &
python LLM/whisper_ros.py --model "base" &

