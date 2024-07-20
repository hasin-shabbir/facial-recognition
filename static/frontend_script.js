function initializePageText(mode){
    const pageTitle = document.getElementById("pageTitle");
    const startButton = document.getElementById("startButton");
    const otherButton = document.getElementById('otherButton');
    if (mode === 'register') {
        pageTitle.textContent = "Register";
        startButton.innerText = "Register";
        otherButton.innerHTML = 'Login instead';
        otherButton.onclick = () => handleClick('Login');
    } else {
        pageTitle.textContent = "Login";
        startButton.innerText = "Login";
        otherButton.innerHTML = 'Don\'t have an account? Register here';
        otherButton.onclick = () => handleClick('Register');
    }
}
function initializePage(mode, username){
    const baseUrl = (window.location.origin !== undefined) && (window.location.origin !== null) && (window.location.origin !== "null") ? window.location.origin : 'localhost:8003';
    const socketUrl = mode === 'register' ? `ws://${baseUrl}/register` : `ws://${baseUrl}/login`;
    const socket = new WebSocket(socketUrl);
    const constraints = { video: true };

    document.getElementById("challenge").innerText = "Detecting face...";
    
    let mediaRecorder;
    let currentChallenge;
    let videoStream;
    
    // Handle WebSocket connection opening
    socket.onopen = async function () {
        videoStream = await navigator.mediaDevices.getUserMedia(constraints);
        mediaRecorder = new MediaRecorder(videoStream);
            
        // Start capturing video frames
        captureFrames();
    };
    
    // Handle messages from the backend
    socket.onmessage = function (event) {
        const data = JSON.parse(event.data);
        currentChallenge = data.current_challenge;
        const challenge = document.getElementById("challenge");
        const result = document.getElementById("result");
        
        const loginSuccess = data.success;
        if (loginSuccess === true) {
            result.textContent = data.msg;
            return;
        }
        else if (loginSuccess === false) {
            result.textContent = data.msg;
            return;
        }
        
        challenge.textContent = `Please ${currentChallenge?.replace("_", " ")}`;
    };
    
    // Capture and send video frames
    async function captureFrames() {
        const video = document.getElementById("videoElement");
        video.srcObject = videoStream;
        video.play();
        
        const canvas = document.getElementById("canvasElement");
        const context = canvas.getContext("2d");
        
        setInterval(() => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            const imageData = canvas.toDataURL("image/jpeg");
            data = {
                username: username,
                image: imageData,
            };
            socket.send(JSON.stringify(data));
        }, 500); // Send frames every 500ms
    }
    
    // Cleanup media streams and WebSocket
    function stopMediaTracks(stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    
    socket.onerror = function (error) {
        console.error('WebSocket error:', error);
        cleanup();
    };
    
    socket.onclose = function () {
        console.log('WebSocket connection closed');
        cleanup();
    };
    
    function cleanup() {
        if (mediaRecorder && mediaRecorder.state !== "inactive") {
            mediaRecorder.stop();
        }
        if (videoStream) {
            stopMediaTracks(videoStream);
        }
        document.getElementById("pageTitle").style.display = "none";
        document.getElementById("videoElement").style.display = "none";
        document.getElementById("canvasElement").style.display = "none";
        document.getElementById("challenge").style.display = "none";
        document.getElementById("result").style.display = "block";
        document.getElementById("home-btn").style.display = "block";
    }
    
    window.onbeforeunload = function () {
        cleanup();
        if (socket.readyState === WebSocket.OPEN) {
            socket.close();
        }
    };
}
