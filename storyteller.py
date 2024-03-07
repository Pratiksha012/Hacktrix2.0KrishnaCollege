import asyncio
import streamlit as st
from openai import OpenAI
from hume import HumeStreamClient
from hume.models.config import LanguageConfig
from face import capture_emotion



client = OpenAI(api_key="your_api_key")

# Set page title and favicon
st.set_page_config(page_title="Nexur.ai", page_icon=":book:")

# Add a header and description
st.title("Nexur.ai")
st.markdown("Enter a prompt and select the emotion you want the story to convey. Click 'Generate Story' to see the magic!")

# User input for prompt and story title
user_input_prompt = st.text_area("Write a prompt for the AI storyteller:")
user_input_title = st.text_input("Enter a title for the story:")

# Dropdown for selecting emotion
emotion_options = ["Happy", "Sad", "Excited", "Mysterious", "Surprised", "Angry", "Calm", "Peaceful", "Confused", "Energetic", "Hopeful", "Curious", "Proud", "Amused", "Optimistic", "Determined", "Anxious", "Silly", "Grateful", "Bittersweet", "Frustrated", "Inspired", "Lonely", "Playful", "Serious"]

selected_emotion = st.selectbox("Select the emotion for the story:", emotion_options)


# Button to capture emotion
if st.button("Capture Emotion"):
    st.info("Capturing emotion... Please wait.")
    
    
    # Capture emotion and send to HumeStreamClient
    emotion_result = asyncio.run(capture_emotion())
    
    
    
    all_emotions = emotion_result['face']['predictions'][0]['emotions']

    st.write(all_emotions)
    
    # Sort emotions based on scores in descending order
    sorted_emotions = sorted(all_emotions, key=lambda x: x['score'], reverse=True)
    
    # Extract the names of the top 5 emotions
    top_emotions = [emotion['name'] for emotion in sorted_emotions[:5]]
    
    # Display the result
    st.success(f"Top 5 emotions with high scores: {', '.join(top_emotions)}")

    # Save the top 5 emotions in session state for later use
    st.session_state.top_emotions = top_emotions





# Button to generate the story, images, and audio
if st.button("Generate Story"):
    if user_input_prompt:
        try:
            # Modify the prompt based on the selected emotion
            prompt_with_emotion = f"Write a {selected_emotion.lower()} story: {user_input_prompt} with {', '.join(st.session_state.top_emotions)}"
            
            # Generate story using OpenAI GPT-3.5-turbo-instruct engine
            response = client.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prompt_with_emotion,
                max_tokens=1000
            )
            
            # Store the generated story in session state
            st.session_state.generated_story = response.choices[0].text
            
            # Display the generated story with title
            st.subheader(f"**{user_input_title}**")
            st.write(response.choices[0].text)
            
            generated_text = response.choices[0].text
            words_per_chunk = 100
            
            # Split the generated text into chunks of 50 words
            chunks = [generated_text[i:i + words_per_chunk] for i in range(0, len(generated_text), words_per_chunk)]

            image_number = 1
            
            st.subheader("Generated Images:")
            # Use DALL·E to generate images based on each chunk
            for i, chunk in enumerate(chunks):
                if i >= 5:
                    break
                image_response = client.images.generate(
                    model="dall-e-2",
                    prompt=chunk,
                    n=1,  # Generating one image for each chunk
                    size="256x256"
                )
                
                # Display all the generated images in a column
                for i, image_data in enumerate(image_response.data):
                    image_url = image_data.url
                    st.image(image_url, caption=f"Image {image_number}", use_column_width=True)
                    image_number += 1
            
            # Audio Generation
            if hasattr(st.session_state, 'generated_story'):
                try:
                    speech_response = client.audio.speech.create(
                        model="tts-1",
                        voice="echo",
                        input=st.session_state.generated_story
                    )
                    st.audio(speech_response.content, format="audio/mp3", start_time=0)
                except Exception as e:
                    st.error(f"An error occurred during text-to-speech: {str(e)}")
            
            # Set a flag to indicate that text has been generated
            st.session_state.text_generated = True
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a prompt for the AI storyteller.")


st.markdown("---")
st.markdown("Built with :heart: by Nexus")

st.markdown(
    """
    <style>
        body {
            color: #394240;
            background-color: #F5F5F5;
        }
        .st-bd {
            padding: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)