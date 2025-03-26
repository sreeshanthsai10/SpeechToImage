import speech_recognition as sr
from translate import Translator
from monsterapi import client
import requests
from PIL import Image
from langdetect import detect  

api_key = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VybmFtZSI6ImU1YWNiYjVhZjRhZTdkYjExNjhkYjYwZGMzMWIxZDg3IiwiY3JlYXRlZF9hdCI6IjIwMjQtMDctMjZUMTM6MTE6NTguMzI2MDc3In0.IDslWwiRwH_lY4F6QSFOXKHqqTag3T3813CoZgRA9A4'
monster_client = client(api_key)

recognizer = sr.Recognizer()

with sr.Microphone() as source:
    print("Say something in any language...")
    recognizer.adjust_for_ambient_noise(source)
    audio = recognizer.listen(source)

    try:
        # Recognize speech without specifying language (auto-detects)
        text = recognizer.recognize_google(audio)
        print("You said:", text)
        
        # Detect the language of the input text
        try:
            detected_lang = detect(text)

            # Translate to English
            translator = Translator(from_lang=detected_lang, to_lang="en")
            translated_text = translator.translate(text)
                       
            # Generate image
            model = 'txt2img'
            negprompt = (
                 "low quality, bad anatomy, worst quality, low resolution, "
                 "extra fingers, blurry, ugly, distorted, disfigured, "
                 "bad hands, error, missing fingers, extra digit, "
                 "fewer digits, cropped, jpeg artifacts, signature, "
                 "watermark, username, deformed, text, out of focus, "
                 "long neck, mutated hands, poorly drawn face, mutation, "
                 "disfigured, blurry, bad proportions, cloned face, "
                 "glitchy, double torso, extra arms, extra legs, "
                 "malformed limbs, missing arms, missing legs, "
                 "extra limbs, fused fingers, too many fingers, "
                 "unclear eyes, long body, bad composition, "
                 "bad perspective, unnatural pose, unnatural lighting, "
                 "dull, flat, overexposed, underexposed, grainy, "
                 "b&w, monochrome, 3D render, cartoon, anime, "
                 "doll, CGI, unrealistic, surreal,painting, drawing, sketch, digital art,non realistic"
                ) 
            input_data = {
                'prompt': translated_text,
                'negprompt':negprompt,
                'samples': 1,
                'steps': 50,
                'aspect_ratio': 'square',
                'guidance_scale': 7.5,
                'seed': 2414,
            }

            print("Generating image....")
            result = monster_client.generate(model, input_data)

            img_url = result['output'][0]
            file_name = "image.png"

            response = requests.get(img_url)
            if response.status_code == 200:
                with open(file_name, 'wb') as file:
                    file.write(response.content)
                    print("Image downloaded successfully")

                    img = Image.open(file_name)
                    img.show()
            else:
                print("Failed to download the image")
                
        except Exception as e:
            print(f"Translation error: {e}")
            # If translation fails, use the original text (assuming it's in English)
            print("Using original text for image generation")
            translated_text = text

    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
