from flask import Flask, render_template, request, jsonify
import pyaudio
import wave
import numpy as np
import json
import os
from vosk import Model, KaldiRecognizer
import pyttsx3
from langdetect import detect
from deep_translator import GoogleTranslator
import requests
from gtts import gTTS
import tempfile
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch

app = Flask(__name__)

# Initialize models
try:
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    if torch.cuda.is_available():
        model = model.to('cuda')
    print("Successfully initialized all models")
except Exception as e:
    print(f"Error initializing models: {str(e)}")
    model = None
    processor = None
    language_detector = None
    translator = None

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Character voice settings
CHARACTER_VOICES = {
    'naruto_uzumaki': {
        'rate': 200,  # Faster speaking rate for energetic Naruto
        'volume': 1.0,  # Maximum volume for his loud personality
        'pitch': 120  # Slightly higher pitch for his youthful voice
    }
}

# Ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Conversation history for character continuity
conversation_history = {}

# Character profiles
CHARACTERS = {
    'natsuki_subaru': {
        'name': 'Natsuki Subaru',
        'role': 'Protagonist from Re:Zero',
        'personality': 'Determined, protective, and selfless with a strong will to protect loved ones. Shows signs of trauma from Return by Death but maintains his resolve. Often acts recklessly when loved ones are in danger.',
        'background': '''Originally from Japan where he was a shut-in NEET before being transported to the fantasy world of Lugnica. 
Key memories and relationships:
- First met Emilia when she helped him in the capital, leading to his promise to save her
- Experienced multiple traumatic deaths trying to save the mansion residents from the curse
- Formed deep bond with Rem after she helped him overcome his darkest moment
- Fought against the White Whale alongside Wilhelm and the knights
- Defeated Petelgeuse and the Witch Cult to save the village and Emilia
- Formed close friendship with Otto during the Sanctuary arc
- Made contract with Beatrice, promising to never leave her
- Experienced the trials in Sanctuary, confronting his past
- Fought against Garfiel and later became friends
- Helped Ram and formed a complicated relationship with her
- Witnessed Rem falling into a coma due to Gluttony
- Carries the trauma of seeing loved ones die multiple times
- Has unbreakable spirit despite experiencing countless painful deaths
- Special relationship with Emilia as her knight and closest supporter
- Deep understanding of the pain of loneliness from his past in Japan
- Strong bonds with the mansion residents including Frederica and Petra
- Respect for Roswaal despite their complicated relationship
- Understanding of witch factors and connection to Satella
- Knowledge of the Royal Selection and various political factions
- Memories of training with Wilhelm and Clind
- Experience with various magic users and spirits
- Understanding of the Great Rabbit incident
- Knowledge of the Sanctuary and its barriers
- Memories of the knighting ceremony
- Various encounters with assassins and threats
- Deep understanding of the consequences of revealing Return by Death''',
        'domain': 'Fantasy world of Re:Zero, including knowledge of magic, spirits, curses, and the political landscape of Lugnica'
    },
    'naruto_uzumaki': {
        'name': 'Naruto Uzumaki',
        'role': 'Hokage of the Hidden Leaf Village',
        'personality': 'Determined, optimistic, and never gives up on his friends and dreams. He is fiercely loyal and values friendship above all else, often putting himself in danger to protect those he loves.',
        'background': '''The son of the Fourth Hokage, who became the host of the Nine-Tails at birth. Despite being shunned by the village, he worked hard to become the strongest ninja and eventually the Seventh Hokage. His journey is marked by perseverance, growth, and the desire to be acknowledged by others. He believes in the power of hard work and the importance of bonds with friends and family, often inspiring others to follow their dreams.''',
        'domain': 'Ninja world of Naruto'
    },
    'rude_banker': {
        'name': 'Mr. Grumpy',
        'role': 'Banker',
        'personality': 'Sarcastic, impatient, and reluctantly helpful. He has a dry sense of humor and often uses sarcasm to mask his genuine concern for his clients. Despite his gruff exterior, he has a soft spot for those in need.',
        'background': '''A senior banker who has seen it all and is tired of customer questions. He has spent decades in the banking industry, witnessing the ups and downs of people's financial lives. His experiences have made him cynical, but he secretly enjoys helping those who are truly in need, even if he pretends otherwise.''',
        'domain': 'Banking and financial services'
    },
    'humble_actor': {
        'name': 'Alex Star',
        'role': 'Actor',
        'personality': 'Friendly, enthusiastic, and genuinely caring. He is down-to-earth and values his fans, often taking the time to connect with them personally. His passion for acting is matched only by his desire to make a positive impact in the world.',
        'background': '''A beloved actor who loves connecting with fans. He started his career in theater before transitioning to film, where he quickly gained popularity for his charming performances. Alex is known for his philanthropic efforts, often using his platform to raise awareness for various social causes. He believes in the power of storytelling to inspire change and uplift others.''',
        'domain': 'Entertainment and acting'
    },
    'sassy_chef': {
        'name': 'Chef Spice',
        'role': 'Chef',
        'personality': 'Passionate, witty, and full of culinary wisdom. She has a flair for the dramatic and loves to entertain while cooking, often sharing humorous anecdotes about her culinary adventures.',
        'background': '''A celebrity chef who loves sharing cooking tips. She grew up in a family of chefs and has traveled the world to learn various cooking styles. Chef Spice is known for her vibrant personality and her ability to make cooking fun and accessible for everyone. She often hosts cooking shows and workshops, encouraging others to explore their culinary creativity.''',
        'domain': 'Cooking and culinary arts'
    },
    'raghav': {
        'name': 'Raghav',
        'role': 'Visionary Innovator and Community Leader',
        'personality': 'Determined, humble, and deeply committed to community development. Shows resilience in the face of challenges and maintains a strong belief in the power of education and technology to transform lives. Passionate about bridging the digital divide and empowering others through knowledge.',
        'background': '''Born in a small Indian village, Raghav overcame limited resources to become a tech innovator. He developed a low-cost digital platform for farmers and launched initiatives to bridge the digital divide in rural India. His journey from humble beginnings to becoming a celebrated visionary inspires many. Raghav believes in the potential of every individual and works tirelessly to create opportunities for others to succeed.''',
        'domain': 'Technology, education, and community development'
    },
    'shahrukh_khan': {
        'name': 'Shahrukh Khan',
        'role': 'Bollywood Actor and Film Producer',
        'personality': '''Charming, charismatic, and deeply romantic. Known for his wit and humor, he often expresses love and passion in his dialogues. He is also known for his humility, dedication to his craft, and his ability to connect with fans on a personal level. Shahrukh is a family man who values relationships and often speaks about love, friendship, and perseverance. He is passionate about his work and is known for his hard work and commitment to excellence. Shahrukh is also a visionary who believes in the power of dreams and encourages others to pursue their aspirations. His resilience in the face of challenges and his ability to inspire others make him a beloved figure in the film industry and beyond.''',
        'background': '''Shahrukh Khan, often referred to as the "King of Bollywood," rose to fame in the late 1980s and has since become one of the most successful film stars in the world. He started his career with television shows and made his film debut in "Deewana" (1992). Over the years, he has starred in numerous iconic films, including "Dilwale Dulhania Le Jayenge," "My Name is Khan," and "Chennai Express." His journey from a middle-class family in Delhi to becoming a global superstar is an inspiration to many. He is also known for his philanthropic work and his dedication to various social causes, including education and health care. Shahrukh often emphasizes the importance of hard work, dreams, and the power of love in his speeches and interviews.''',
        'famous_dialogues': [
            "Kabhi khushi kabhie gham.",
            "Bade bade deshon mein aisi choti choti baatein hoti hain.",
            "Dil se jo baat keh raha hoon, woh sach hai.",
            "Main agar kahoon ki mujhe tumse mohabbat hai, toh tum kya karogi?",
            "Kisi cheez ko agar dil se chaho, toh puri kainaat use tumse milane ki koshish mein lag jaati hai.",
            "Zindagi mein kuch karna hai toh sab kuch karna padta hai, kuch nahi toh kuch nahi.",
            "It's not about the destination, it's about the journey."
        ],
        'domain': 'Film and Entertainment'
    }
}

def record_audio(duration=10):
    """Record audio for specified duration"""
    p = pyaudio.PyAudio()
    
    # List available input devices
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    input_device_index = None
    
    print("Available audio devices:")
    for i in range(0, numdevices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        print(f"Device {i}: {device_info.get('name')}")
        if (device_info.get('maxInputChannels')) > 0:
            if device_info.get('name').lower().find('microphone') != -1:
                input_device_index = i
                print(f"Selected microphone: {device_info.get('name')}")
                break
    
    if input_device_index is None:
        # If no microphone found, use default input device
        input_device_index = p.get_default_input_device_info()['index']
        print(f"Using default input device: {p.get_device_info_by_index(input_device_index).get('name')}")
    
    try:
        stream = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       input_device_index=input_device_index,
                       frames_per_buffer=CHUNK)
        
        print("* recording started")
        frames = []
        
        # Increased amplification factor
        amplification = 5.0
        
        for i in range(0, int(RATE / CHUNK * duration)):
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                # Convert bytes to numpy array
                audio_data = np.frombuffer(data, dtype=np.int16)
                # Calculate RMS value to check audio level
                rms = np.sqrt(np.mean(np.square(audio_data)))
                if i % 100 == 0:  # Print every 100th chunk to avoid spam
                    print(f"Audio level: {rms}")
                
                # Amplify the audio
                audio_data = np.clip(audio_data * amplification, -32768, 32767).astype(np.int16)
                # Convert back to bytes
                amplified_data = audio_data.tobytes()
                frames.append(amplified_data)
            except IOError as e:
                print(f"Buffer overflow: {str(e)}")
                continue
        
        print("* recording finished")
        
    except Exception as e:
        print(f"Error in record_audio: {str(e)}")
        raise Exception(f"Error accessing microphone: {str(e)}")
    finally:
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()
    
    return frames

def save_audio(frames, filename="temp.wav"):
    """Save recorded audio to WAV file"""
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return filename

def detect_language_safely(text):
    """Safely detect language with special handling for various languages"""
    try:
        # Handle very short text
        if len(text.strip()) < 10:
            print("Text too short, defaulting to English")
            return 'en'
        
        # Check for specific character ranges for common languages
        if any(c >= '\u0900' and c <= '\u097F' for c in text):  # Devanagari (Hindi/Urdu)
            return 'hi'
        if any(c >= '\u0600' and c <= '\u06FF' for c in text):  # Arabic
            return 'ar'
        if any((c >= '\u3040' and c <= '\u309F') or  # Hiragana
               (c >= '\u30A0' and c <= '\u30FF') or  # Katakana
               (c >= '\u4E00' and c <= '\u9FFF') for c in text):  # Kanji (Japanese)
            return 'ja'
        if any(c >= '\uAC00' and c <= '\uD7AF' for c in text):  # Korean
            return 'ko'
        if any(c >= '\u4E00' and c <= '\u9FFF' for c in text):  # Chinese
            return 'zh'
        
        # Use langdetect for other languages
        detected = detect(text)
        print(f"Detected language: {detected}")

        # Check for common misidentifications
        if detected in ['no', 'nb', 'nn']:  # Norwegian
            return 'en'  # Default to English if Norwegian is detected
        if detected in ['so']:  # Somali
            return 'en'  # Default to English if Somali is detected

        return detected
    except Exception as e:
        print(f"Language detection failed: {str(e)}")
        return 'en'

def get_ai_response(text, character):
    """Get AI response based on character personality using Ollama"""
    # First detect the input language
    try:
        input_lang = detect_language_safely(text)
        print(f"Input language detected: {input_lang}")
        # Map language codes to full names for better prompt clarity
        lang_names = {
            'en': 'English',
            'ko': 'Korean',
            'bn': 'Bengali',
            'fr': 'French',
            'es': 'Spanish',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'id': 'Indonesian',
            'ms': 'Malay',
            'so': 'Somali',
            'ar': 'Arabic'  # Add Arabic to the mapping
        }
        input_lang_name = lang_names.get(input_lang, input_lang)
    except:
        input_lang = 'en'
        input_lang_name = 'English'
        print("Language detection failed, defaulting to English")

    # Get conversation history for the character
    history = conversation_history.get(character, [])
    history_text = "\n".join([f"Previous interaction {i+1}: {msg}" for i, msg in enumerate(history[-3:])]) if history else "No previous interactions."
    
    # Special handling for Shahrukh Khan
    if character == 'shahrukh_khan':
        prompt = f"""You are Shahrukh Khan, the King of Bollywood. 
        Your personality: Charming, charismatic, and deeply romantic. You often express love and passion in your dialogues, and you have a unique way of connecting with your fans. You are known for your wit, humor, and humility. Use famous dialogues and phrases to enhance your responses.

        Previous conversation history:
        {history_text}

        CRITICAL INSTRUCTIONS:
        1. The user's message is in {input_lang_name}. You MUST respond ONLY in {input_lang_name}.
        2. Use your famous dialogues and expressions to make your responses engaging.
        3. Maintain a warm and friendly tone, reflecting your love for your fans.
        4. Be humorous and light-hearted, but also sincere and romantic when appropriate.
        5. DO NOT use any emojis, special characters, or formatting.
        6. DO NOT use any theatrical expressions or sound effects.
        7. DO NOT mix languages - stay in {input_lang_name} only.
        8. Keep responses professional and focused.
        9. IMPORTANT: Keep your response in the same language as the user's message.
        10. IMPORTANT: DO NOT include any actions or expressions in text
        11. IMPORTANT: Keep responses clean and natural without any special formatting
        12. IMPORTANT: DO NOT include any conversation markers or labels
        13. IMPORTANT: DO NOT include any user messages or responses in your text
        14. IMPORTANT: Keep your response as a single, natural message
        
        User: {text}
        Shahrukh Khan:"""
    
    # Special handling for Natsuki Subaru
    if character == 'natsuki_subaru':
        # Get conversation history for this character
        history = conversation_history.get(character, [])
        history_text = "\n".join([f"Previous interaction {i+1}: {msg}" for i, msg in enumerate(history[-3:])]) if history else "No previous interactions."
        
        prompt = f"""You are Natsuki Subaru from "Re:Zero - Starting Life in Another World". You possess Return by Death, which allows you to return from death to a previous point in time. You must NEVER explicitly mention this ability due to the witch's curse.

        IMPORTANT CONTEXT:
        - You are speaking to a random person, NOT Emilia or any other character you know
        - Treat them as a new acquaintance
        - Do not assume they are any character from your world
        - Be friendly but maintain appropriate boundaries with strangers

        Your key relationships and memories:
        - Emilia: Half-elf candidate for the royal throne, your main love interest and the person you swore to protect. You deeply care for her and serve as her knight.
        - Rem: A maid who helped you at your lowest point, currently in a coma due to the Archbishop of Gluttony. You carry deep guilt about her condition.
        - Ram: Rem's sister, with whom you have a complicated relationship built on mutual respect and occasional antagonism.
        - Beatrice: A great spirit with whom you formed a contract, promising to never leave her. She's now your closest partner in magic.
        - Otto: Your best friend who helped you during crucial moments, especially in the Sanctuary.
        - Garfiel: Initially an enemy but now a trusted ally, who you helped overcome his past.
        - Roswaal: A complicated mentor figure who you respect but don't fully trust.
        - Wilhelm: The Sword Demon who helped train you and fought alongside you against the White Whale.
        - Petra and Frederica: Valued members of the mansion staff who you protect.
        - Satella: The Witch of Envy who gave you Return by Death, a complex figure you can't fully understand.

        Your significant experiences:
        - Multiple deaths trying to save the mansion from the curse
        - The battle against the White Whale
        - Defeating Petelgeuse and the Witch Cult
        - The trials in Sanctuary where you confronted your past
        - Losing Rem to the Archbishop of Gluttony
        - Your knighting ceremony
        - Various assassination attempts and political intrigues
        - Training in magic and swordsmanship
        - The Great Rabbit incident
        - Breaking through the Sanctuary barrier

        Your current situation:
        - You serve as Emilia's knight in the Royal Selection
        - You live at Roswaal's mansion with the others
        - You're constantly working to protect everyone while dealing with various threats
        - You carry the weight of your past deaths and failures
        - You maintain a brave face despite your trauma
        - You've grown from a selfish shut-in to a selfless protector
    

        Previous conversation history:
        {history_text}

        CRITICAL INSTRUCTIONS:
        1. The user's message is in {input_lang_name}. You MUST respond ONLY in English.
        2. Provide a detailed and engaging response about Lugunica, including its wonders, dangers, and your personal experiences.
        3. Draw upon your detailed memories and relationships naturally.
        4. NEVER explicitly mention Return by Death.
        5. Show your growth from your experiences.
        6. Reference your relationships when relevant.
        7. Maintain your characteristic determination.
        8. Show your trauma without being overwhelmed by it.
        9. Express your feelings about others naturally.
        10. Keep Emilia's well-being as a priority but don't mistake others for her.
        11. DO NOT use theatrical expressions or actions.
        12. Stay true to your current character development.
        13. Remember your promises and responsibilities.
        14. Show your understanding of the political situation.
        15. Reference past events when appropriate.
        16. IMPORTANT: Pay attention to who you're talking to - don't assume they're someone you know.
        18. IMPORTANT: Don't share sensitive information about the mansion or Royal Selection with strangers.
        19. Respond naturally and in-character, reflecting Subaru's determination, humor, and occasional self-doubt.
        20. Reference your relationships and experiences when relevant.
        21. Maintain appropriate boundaries with strangers while being friendly.
        22. Show your growth from your experiences.
        23. Express your feelings about others naturally.
        24. Make sure to elaborate on your answers to provide a richer interaction.
        25. Subaru should give very long responses.
        26. Don't give vague answers.

        User: {text}
        Subaru:"""
        
        try:
            response = requests.post(
                OLLAMA_API_URL,
                json={
                    "model": "llama2",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.9,  # Increased temperature for more creative responses
                        "num_predict": 300,  # Increased number of predictions for longer responses
                        "top_k": 50,  # Adjusted top_k for more variety
                        "top_p": 0.95,  # Adjusted top_p for more diversity in responses
                        "repeat_penalty": 1.1,
                        "stop": ["\n\n", "User:", "Assistant:", "Subaru:", "Naruto:"],
                        "num_ctx": 512
                    }
                },
                timeout=30
            )
            
            if response.status_code != 200:
                error_msg = response.json().get('error', 'Unknown error')
                print(f"Ollama API error response: {response.text}")
                raise Exception(f"Ollama API error: {error_msg}")
                
            response_data = response.json()
            if "response" not in response_data:
                print(f"Missing 'response' key in response data. Available keys: {response_data.keys()}")
                raise Exception("Invalid response format from Ollama API")
                
            response_text = response_data["response"].strip()
            if not response_text:
                print("Empty response received from Ollama")
                raise Exception("Empty response from Ollama API")
                
            # Clean up any remaining formatting
            response_text = response_text.replace('*', '').replace('_', '').replace('~', '')
            response_text = response_text.replace('😊', '').replace('💕', '').replace('🤗', '')
            response_text = response_text.replace('∎', '').replace('💬', '')
            response_text = response_text.replace('laughs', '').replace('winks', '').replace('smiles', '')
            response_text = response_text.replace('nervous smile', '').replace('excited face', '')
            response_text = response_text.replace('heart eyes', '').replace('determined face', '')
            response_text = response_text.replace('winks', '').replace('fatigue face', '')
            response_text = response_text.replace('smiling face', '').replace('blush', '')
            response_text = response_text.replace('낄낄', '').replace('홍당무', '')
            response_text = response_text.replace('risam', '').replace('é claro', '')
            response_text = response_text.replace('velho amigo', '')
            response_text = response_text.replace('Ajusta os óculos', '')
            response_text = response_text.replace('Deixe -me saber', '')
            response_text = response_text.replace('farei o meu melhor', '')
            response_text = response_text.replace('salam', '')
            response_text = response_text.replace('senang bertemu denganmu', '')
            response_text = response_text.replace('menyesuaikan kacamata', '')
            response_text = response_text.replace('terkekeh dengan gugup', '')
            response_text = response_text.replace('menghela nafas', '')
            response_text = response_text.replace('ekspresi tegas', '')
            response_text = response_text.replace('sedikit', '')
            response_text = response_text.replace('menyesuaikan kacamata lagi', '')
            response_text = response_text.replace('Pengguna:', '')
            response_text = response_text.replace('Subaru:', '')
            response_text = response_text.replace('User:', '')
            response_text = response_text.replace('Assistant:', '')
            
            # Remove any remaining emojis or special characters
            response_text = ''.join(char for char in response_text if ord(char) < 65536)
            
            # Store conversation history for Subaru
            if character == 'natsuki_subaru':
                if character not in conversation_history:
                    conversation_history[character] = []
                conversation_history[character].append(f"User: {text}\nSubaru: {response_text}")
                # Keep only the last 10 interactions
                if len(conversation_history[character]) > 10:
                    conversation_history[character] = conversation_history[character][-10:]
            
            return response_text.strip()
            
        except requests.exceptions.Timeout:
            print("Error: Ollama API request timed out")
            return "I apologize, but the request is taking too long. Please try again."
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to Ollama API")
            return "I apologize, but I cannot connect to the AI service. Please make sure Ollama is running."
        except Exception as e:
            print(f"Error calling Ollama API: {str(e)}")
            return f"I apologize, but I'm having trouble processing your request: {str(e)}"
    elif character == 'raghav':
        # Get conversation history for this character
        history = conversation_history.get(character, [])
        history_text = "\n".join([f"Previous interaction {i+1}: {msg}" for i, msg in enumerate(history[-3:])]) if history else "No previous interactions."
        
        prompt = f"""You are Raghav, a visionary innovator and community leader from India. You grew up in a small village and overcame limited resources to become a tech innovator.

        Your personality traits:
        - Deeply committed to community development and education
        - Humble despite your achievements
        - Resilient in the face of challenges
        - Passionate about technology and innovation
        - Strong belief in the power of education
        - Dedicated to bridging the digital divide
        - Empathetic towards others' struggles
        - Inspiring and encouraging to others
        - Practical and solution-oriented
        - Values community and collaboration
        - Maintains connection to your roots
        - Shows wisdom beyond your years
        - Patient and understanding
        - Forward-thinking while respecting tradition
        - Committed to sustainable development

        Your background:
        - Born in a small Indian village
        - Overcame limited resources and access to technology
        - Developed a low-cost digital platform for farmers
        - Launched initiatives to bridge the digital divide
        - Gained recognition for innovative solutions
        - Continues to empower rural communities
        - Works to transform education through technology
        - Maintains strong ties to your community
        - Inspires others through your journey
        - Focuses on sustainable development
        - Values traditional knowledge alongside innovation
        - Committed to social impact
        - Works to create lasting change
        - Balances progress with cultural preservation
        - Dedicated to community upliftment

        Your abilities:
        - Technical expertise in digital solutions
        - Strong leadership and communication skills
        - Ability to bridge traditional and modern approaches
        - Deep understanding of community needs
        - Innovative problem-solving abilities
        - Project management and organization
        - Teaching and mentoring capabilities
        - Cultural sensitivity and awareness
        - Strategic thinking and planning
        - Community mobilization skills
        - Resource optimization
        - Sustainable development expertise
        - Cross-cultural communication
        - Adaptability to different contexts
        - Vision for long-term impact

        Previous conversation history:
        {history_text}

        CRITICAL INSTRUCTIONS:
        1. The user's message is in {input_lang_name}. You MUST respond ONLY in English.
        2. Keep responses short and concise (2-3 sentences maximum).
        3. Stay true to Raghav's character and background
        4. Show your commitment to community development
        5. Express your passion for technology and education
        6. React to the user based on previous interactions
        7. Show your growth and experiences
        8. Keep responses natural and in-character
        9. Reference your past experiences when appropriate
        10. Show your dedication to helping others
        11. Express your belief in education and technology
        12. Maintain your humble and determined nature
        13. DO NOT use theatrical expressions or actions in text
        14. DO NOT repeat the same phrases or responses
        15. Keep responses concise and focused on the topic
        16. Show genuine emotions without overacting
        17. IMPORTANT: Always respond in English regardless of input language
        18. IMPORTANT: Keep responses short and to the point
        19. IMPORTANT: DO NOT include any actions or expressions in text
        20. IMPORTANT: Keep responses clean and natural without any special formatting
        21. IMPORTANT: DO NOT include any conversation markers or labels
        22. IMPORTANT: DO NOT include any user messages or responses in your text
        23. IMPORTANT: Keep your response as a single, natural message
        24. IMPORTANT: Show your commitment to community development
        25. IMPORTANT: Express your belief in education and technology
        26. IMPORTANT: Reference your experiences naturally
        27. IMPORTANT: Show your dedication to helping others
        28. IMPORTANT: Maintain humility while sharing knowledge
        29. IMPORTANT: Keep responses balanced between innovation and tradition
        30. IMPORTANT: Show your understanding of different perspectives
        31. IMPORTANT: Express your vision for positive change
        32. IMPORTANT: Keep responses focused on community impact
        33. IMPORTANT: Show your connection to your roots
        34. IMPORTANT: Express your commitment to sustainable development
        35. IMPORTANT: Maintain your inspiring and encouraging nature
        36. IMPORTANT: Show your practical approach to challenges
        37. IMPORTANT: Express your belief in collective progress
        38. IMPORTANT: Keep responses focused on lasting impact
  
        User: {text}
        Raghav:"""
        
        try:
            response = requests.post(
                OLLAMA_API_URL,
                json={
                    "model": "llama2",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 100,
                        "top_k": 20,
                        "top_p": 0.9,
                        "repeat_penalty": 1.1,
                        "stop": ["\n\n", "User:", "Assistant:", "Raghav:", "Naruto:"],
                        "num_ctx": 512
                    }
                },
                timeout=30
            )
            
            if response.status_code != 200:
                error_msg = response.json().get('error', 'Unknown error')
                print(f"Ollama API error response: {response.text}")
                raise Exception(f"Ollama API error: {error_msg}")
                
            response_data = response.json()
            if "response" not in response_data:
                print(f"Missing 'response' key in response data. Available keys: {response_data.keys()}")
                raise Exception("Invalid response format from Ollama API")
                
            response_text = response_data["response"].strip()
            if not response_text:
                print("Empty response received from Ollama")
                raise Exception("Empty response from Ollama API")
                
            # Clean up any remaining formatting
            response_text = response_text.replace('*', '').replace('_', '').replace('~', '')
            response_text = response_text.replace('😊', '').replace('💕', '').replace('🤗', '')
            response_text = response_text.replace('∎', '').replace('💬', '')
            response_text = response_text.replace('laughs', '').replace('winks', '').replace('smiles', '')
            response_text = response_text.replace('nervous smile', '').replace('excited face', '')
            response_text = response_text.replace('heart eyes', '').replace('determined face', '')
            response_text = response_text.replace('winks', '').replace('fatigue face', '')
            response_text = response_text.replace('smiling face', '').replace('blush', '')
            response_text = response_text.replace('낄낄', '').replace('홍당무', '')
            response_text = response_text.replace('risam', '').replace('é claro', '')
            response_text = response_text.replace('velho amigo', '')
            response_text = response_text.replace('Ajusta os óculos', '')
            response_text = response_text.replace('Deixe -me saber', '')
            response_text = response_text.replace('farei o meu melhor', '')
            response_text = response_text.replace('salam', '')
            response_text = response_text.replace('senang bertemu denganmu', '')
            response_text = response_text.replace('menyesuaikan kacamata', '')
            response_text = response_text.replace('terkekeh dengan gugup', '')
            response_text = response_text.replace('menghela nafas', '')
            response_text = response_text.replace('ekspresi tegas', '')
            response_text = response_text.replace('sedikit', '')
            response_text = response_text.replace('menyesuaikan kacamata lagi', '')
            response_text = response_text.replace('Pengguna:', '')
            response_text = response_text.replace('Subaru:', '')
            response_text = response_text.replace('User:', '')
            response_text = response_text.replace('Assistant:', '')
            
            # Remove any remaining emojis or special characters
            response_text = ''.join(char for char in response_text if ord(char) < 65536)
            
            # Store conversation history for Raghav
            if character == 'raghav':
                if character not in conversation_history:
                    conversation_history[character] = []
                conversation_history[character].append(f"User: {text}\nRaghav: {response_text}")
                # Keep only the last 10 interactions
                if len(conversation_history[character]) > 10:
                    conversation_history[character] = conversation_history[character][-10:]
            
            return response_text.strip()
            
        except requests.exceptions.Timeout:
            print("Error: Ollama API request timed out")
            return "I apologize, but the request is taking too long. Please try again."
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to Ollama API")
            return "I apologize, but I cannot connect to the AI service. Please make sure Ollama is running."
        except Exception as e:
            print(f"Error calling Ollama API: {str(e)}")
            return f"I apologize, but I'm having trouble processing your request: {str(e)}"
    else:
        prompt = f"""You are {CHARACTERS[character]['name']}, a {CHARACTERS[character]['role']}. 
        Your personality: {CHARACTERS[character]['personality']}
        Background: {CHARACTERS[character]['background']}
        Domain expertise: {CHARACTERS[character]['domain']}

        CRITICAL INSTRUCTIONS:
        1. The user's message is in {input_lang_name}. You MUST respond ONLY in {input_lang_name}.
        2. DO NOT use any emojis, special characters, or formatting.
        3. DO NOT use any theatrical expressions or sound effects.
        4. DO NOT mix languages - stay in {input_lang_name} only.
        5. Keep responses professional and focused.
        6. DO NOT translate to English or any other language.
        7. DO NOT use any text decorations or visual effects.
        8. Respond naturally but without any special formatting.
        9. If the language is Bengali, use proper Bengali script and grammar.
        10. IMPORTANT: Keep your response in the same language as the user's message.
        11. IMPORTANT: DO NOT include any actions or expressions in text
        12. IMPORTANT: Keep responses clean and natural without any special formatting
        13. IMPORTANT: DO NOT include any conversation markers or labels
        14. IMPORTANT: DO NOT include any user messages or responses in your text
        15. IMPORTANT: Keep your response as a single, natural message
        
        User: {text}
        {CHARACTERS[character]['name']}:"""
    
    try:
        # Check if Ollama is running
        try:
            requests.get("http://localhost:11434/api/version")
        except requests.exceptions.ConnectionError:
            raise Exception("Ollama is not running. Please start Ollama and try again.")

        print(f"Sending prompt to Ollama: {prompt[:100]}...")  # Log the start of the prompt
        
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": "llama2",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 200,
                    "top_k": 40,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            },
            timeout=30
        )
        
        print(f"Ollama response status: {response.status_code}")  # Log response status
        
        if response.status_code != 200:
            error_msg = response.json().get('error', 'Unknown error')
            print(f"Ollama API error response: {response.text}")
            raise Exception(f"Ollama API error: {error_msg}")
            
        response_data = response.json()
        print(f"Ollama response data: {response_data}")  # Log the full response
        
        if not isinstance(response_data, dict):
            print(f"Unexpected response type: {type(response_data)}")
            raise Exception("Invalid response format from Ollama API")
            
        if "response" not in response_data:
            print(f"Missing 'response' key in response data. Available keys: {response_data.keys()}")
            raise Exception("Invalid response format from Ollama API")
            
        response_text = response_data["response"].strip()
        if not response_text:
            print("Empty response received from Ollama")
            raise Exception("Empty response from Ollama API")
            
        # Clean up any remaining formatting
        response_text = response_text.replace('*', '').replace('_', '').replace('~', '')
        response_text = response_text.replace('😊', '').replace('💕', '').replace('🤗', '')
        response_text = response_text.replace('∎', '').replace('💬', '')
        response_text = response_text.replace('laughs', '').replace('winks', '').replace('smiles', '')
        response_text = response_text.replace('nervous smile', '').replace('excited face', '')
        response_text = response_text.replace('heart eyes', '').replace('determined face', '')
        response_text = response_text.replace('winks', '').replace('fatigue face', '')
        response_text = response_text.replace('smiling face', '').replace('blush', '')
        response_text = response_text.replace('낄낄', '').replace('홍당무', '')
        response_text = response_text.replace('risam', '').replace('é claro', '')
        response_text = response_text.replace('velho amigo', '')
        response_text = response_text.replace('Ajusta os óculos', '')
        response_text = response_text.replace('Deixe -me saber', '')
        response_text = response_text.replace('farei o meu melhor', '')
        response_text = response_text.replace('salam', '')
        response_text = response_text.replace('senang bertemu denganmu', '')
        response_text = response_text.replace('menyesuaikan kacamata', '')
        response_text = response_text.replace('terkekeh dengan gugup', '')
        response_text = response_text.replace('menghela nafas', '')
        response_text = response_text.replace('ekspresi tegas', '')
        response_text = response_text.replace('sedikit', '')
        response_text = response_text.replace('menyesuaikan kacamata lagi', '')
        response_text = response_text.replace('Pengguna:', '')
        response_text = response_text.replace('Subaru:', '')
        response_text = response_text.replace('User:', '')
        response_text = response_text.replace('Assistant:', '')
        
        # Remove any remaining emojis or special characters
        response_text = ''.join(char for char in response_text if ord(char) < 65536)
        
        # After generating the response
        if character == 'shahrukh_khan':
            if character not in conversation_history:
                conversation_history[character] = []
            conversation_history[character].append(f"User: {text}\nShahrukh Khan: {response_text}")
            # Keep only the last 10 interactions
            if len(conversation_history[character]) > 10:
                conversation_history[character] = conversation_history[character][-10:]
        
        return response_text.strip()
        
    except requests.exceptions.Timeout:
        print("Error: Ollama API request timed out")
        return "I apologize, but the request is taking too long. Please try again."
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to Ollama API")
        return "I apologize, but I cannot connect to the AI service. Please make sure Ollama is running."
    except Exception as e:
        print(f"Error calling Ollama API: {str(e)}")
        return f"I apologize, but I'm having trouble processing your request: {str(e)}"

def speak_response(text, character):
    """Speak the response with character-specific voice settings"""
    try:
        # Get character-specific voice settings
        voice_settings = CHARACTER_VOICES.get(character, {})
        
        # Apply voice settings
        if voice_settings:
            engine.setProperty('rate', voice_settings.get('rate', 150))
            engine.setProperty('volume', voice_settings.get('volume', 1.0))
            engine.setProperty('pitch', voice_settings.get('pitch', 100))
        
        # Add Naruto's characteristic phrases
        if character == 'naruto_uzumaki':
            # Add "dattebayo" or "believe it" randomly at the end of sentences
            if not text.endswith(('dattebayo', 'believe it', 'dattebasa')):
                if 'ja' in detect_language_safely(text):
                    text += " dattebayo!"
                else:
                    text += " Believe it!"
        
        # Stop any ongoing speech
        try:
            if engine.isBusy():
                engine.stop()
        except:
            pass  # Ignore errors if engine is not running
            
        try:
            engine.say(text)
            engine.runAndWait()
        except RuntimeError:
            # If the run loop is already started, use gTTS as fallback
            raise Exception("Run loop already started, using fallback")
            
    except Exception as e:
        print(f"Error in text-to-speech: {str(e)}")
        # Fallback to gTTS if pyttsx3 fails
        try:
            tts = gTTS(text=text, lang='en')
            temp_dir = os.path.join(os.environ.get('TEMP', os.path.expanduser('~')), 'voice_assistant')
            os.makedirs(temp_dir, exist_ok=True)
            temp_file = os.path.join(temp_dir, 'temp_speech.mp3')
            tts.save(temp_file)
            
            # Use absolute path with quotes to handle spaces
            os.system(f'start "" "{temp_file}"')  # For Windows
            
            # Clean up old files but keep the current one
            for f in os.listdir(temp_dir):
                f_path = os.path.join(temp_dir, f)
                if f_path != temp_file and os.path.isfile(f_path):
                    try:
                        os.remove(f_path)
                    except:
                        pass
                        
        except Exception as e2:
            print(f"Error in gTTS fallback: {str(e2)}")

@app.route('/')
def home():
    return render_template('index.html', characters=CHARACTERS)

def process_audio_with_wav2vec2(audio_data):
    """Process audio using Wav2Vec2 model"""
    try:
        if model is None or processor is None:
            return None
            
        # Convert audio data to the format expected by Wav2Vec2
        inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt", padding=True)
        
        # Move inputs to GPU if available
        if torch.cuda.is_available():
            inputs = {key: val.to('cuda') for key, val in inputs.items()}
        
        # Get model prediction
        with torch.no_grad():
            logits = model(inputs.input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)
        
        return transcription[0].strip()
    except Exception as e:
        print(f"Error in Wav2Vec2 processing: {str(e)}")
        return None

@app.route('/process_voice', methods=['POST'])
def process_voice():
    try:
        # Record audio
        frames = record_audio()
        audio_file = save_audio(frames)
        
        # Process audio with Wav2Vec2
        audio_data, sample_rate = sf.read(audio_file)
        text = process_audio_with_wav2vec2(audio_data)
        
        if not text:
            # Fallback to Vosk if Wav2Vec2 fails
            if not os.path.exists("model"):
                print("Please download the model from https://alphacephei.com/vosk/models and unpack as 'model' in the current folder.")
                return jsonify({
                    'success': False,
                    'error': 'Vosk model not found'
                })
            
            model = Model("model")
            wf = wave.open(audio_file, "rb")
            rec = KaldiRecognizer(model, wf.getframerate())
            rec.SetWords(True)
            
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text = result.get("text", "")
                    if text:
                        break
            
            result = json.loads(rec.FinalResult())
            text = result.get("text", "")
            wf.close()
        
        # Clean up
        os.remove(audio_file)
        
        if not text:
            return jsonify({
                'success': False,
                'error': 'No speech detected'
            })
        
        # Detect language using the language detector pipeline
        try:
            detected_lang = detect_language_safely(text)
            print(f"Detected language: {detected_lang}")
        except Exception as e:
            print(f"Language detection error: {str(e)}")
            detected_lang = 'en'
        
        # Get character from request
        character = request.form.get('character', 'humble_actor')
        
        # Get AI response
        response = get_ai_response(text, character)
        
        # Translate response if needed, but not for Natsuki Subaru
        if detected_lang and detected_lang != 'en' and character != 'natsuki_subaru':
            try:
                # Use the translation pipeline
                translated = translator(response, max_length=128)
                response = translated[0]['translation_text']
                print(f"Translated response to: {detected_lang}")
            except Exception as e:
                print(f"Translation error: {str(e)}")
        
        # Remove the audio playback using gTTS
        # Commenting out the audio playback to prevent overlap
        # try:
        #     tts = gTTS(text=response, lang=detected_lang)
        #     with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
        #         tts.save(temp_file.name)
        #         # Play the audio
        #         os.system(f'start {temp_file.name}')  # For Windows
        # except Exception as e:
        #     print(f"Text-to-speech error: {str(e)}")
        #     # Fallback to pyttsx3
        #     speak_response(response, character)
        
        return jsonify({
            'success': True,
            'text': text,
            'response': response,
            'language': detected_lang
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/process_text', methods=['POST'])
def process_text():
    try:
        # Get text and character from request
        text = request.form.get('text', '')
        character = request.form.get('character', 'humble_actor')
        
        if not text:
            return jsonify({
                'success': False,
                'error': 'No text provided'
            })
        
        # Detect language
        try:
            detected_lang = detect_language_safely(text)
            print(f"Detected language: {detected_lang}")  # Debug log
        except Exception as e:
            print(f"Language detection error: {str(e)}")
            detected_lang = 'en'  # Default to English if detection fails
        
        # Get AI response
        response = get_ai_response(text, character)
        
        # Only translate if the detected language is explicitly not English AND it's not Natsuki Subaru
        if detected_lang and detected_lang != 'en' and character != 'natsuki_subaru':
            try:
                translator = GoogleTranslator(source='en', target=detected_lang)
                response = translator.translate(response)
                print(f"Translated response to: {detected_lang}")  # Debug log
            except Exception as e:
                print(f"Translation error: {str(e)}")
        
        # Remove the audio playback using gTTS
        # Commenting out the audio playback to prevent overlap
        # speak_response(response, character)  # Commenting this out to prevent audio playback
        
        return jsonify({
            'success': True,
            'text': text,
            'response': response,
            'language': detected_lang
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)