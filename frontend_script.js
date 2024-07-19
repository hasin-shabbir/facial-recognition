function initializePage(mode){
    const pageTitle = document.getElementById("pageTitle");
    if (mode === 'register') {
        pageTitle.textContent = "Register";
    } else {
        pageTitle.textContent = "Login";
    }

    const socketUrl = mode === 'register' ? 'ws://localhost:8003/register' : 'ws://localhost:8003/login';
    const socket = new WebSocket("ws://localhost:8003/login");
    const constraints = { video: true };
    
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
        const instructions = document.getElementById("instructions");
        
        instructions.textContent = `Please ${currentChallenge.replace("_", " ")}`;
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
            socket.send(imageData);
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
        if (captureInterval) {
            clearInterval(captureInterval);
        }
    }
    
    window.onbeforeunload = function () {
        cleanup();
        if (socket.readyState === WebSocket.OPEN) {
            socket.close();
        }
    };
}
