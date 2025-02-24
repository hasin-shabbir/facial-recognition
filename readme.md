# Facial Recognition User Validation

Also employs liveness/spoof checks through randomized challenges, detection of micromovements, and full-face detection.

## DEMO
(Sad reacts for weird smile/expressions, won't happen again 😤)

https://github.com/user-attachments/assets/8b1f3e8e-cab0-4593-a208-f5cac8e3631f


## How to?
- Register on `/register`
- Login on `/login` using the same username
- Use a wrong username+face combination and (hopefully) it won't work :')

## Todos (feel free to suggest smart/scalable solutions):
- Deepfake detection
- Depth+texture detection as an additional check for liveness (can we even do those without additional cameras etc.?)
- Checks for continuity among frames (adds robustness against spoofing attacks)
- This is just an MVP using quick math. Challenges can be improved in terms of performance etc. using better math but YOLO 🤙🏽
- Try different frame communication rates (though values between 500ms to 750ms seem to work best) in terms of cost, performance etc.

## To run locally using Docker
- uncomment `ENV DATABASE_URL` in dockerfile and set it to your own url (if you are a friend or from Noon, hit me up and I'll provide you with one already set up)
- `docker build -t facial-recognition .`
- `docker run facial-recognition`
- Navigate to localhost on the indicated port

## Run locally without Docker
- Install cmake (for example `brew install cmake`)
- `pip install --no-cache-dir -r requirements.txt`
- create a `.env` file in the root directory (see env.example) and set `DATABASE_URL = {your_db_url}` (again, if you are a friend or from Noon, hit me up and I'll provide you with one already set up)
- `uvicorn main:app --reload --port 8003 --env-file .env`
- Navigate to `http://localhost:8003`

## Report an error or a bug
- Open an Issue

## I have a suggestion/improvement
- Open a PR

## I have a suggestion/improvement but don't know how to code
- Open an Issue

## FAQs
### Why does the build process take so long?
Well, there's currently a dependency on dlib through face_recognition and building its wheel takes forever. We could do either:
- Use pre-built wheel stored somewhere
- Wait for me to use some alternative lib/model. I just built an MVP to show that stuff works.

### Getting an error that says image type should be 8-bit or RGB etc.
- Use numpy version <=2.0 (I'm using 1.26.4 and it works). v>=2.0 breaks stuff for some libs etc.

### Cannot build wheel for dlib
- Ensure you have `cmake` installed in your system e.g. using `brew install cmake`. Generally, make sure you have everything mentioned under `# Install system dependencies` in dockerfile installed in your system/venv. I'll do dependency cleanup later and update.
