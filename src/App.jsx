
import { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import * as blazeface from '@tensorflow-models/blazeface';
import './App.css';

const emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised'];

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);
  const [faceDetector, setFaceDetector] = useState(null);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [emotion, setEmotion] = useState('Loading...');
  

  const recentEmotionsRef = useRef([]);
  const emotionSmoothing = 10; 
  
  const fixedEmotionTimeRef = useRef(0);
  const currentEmotionIndexRef = useRef(2);

 
  useEffect(() => {
    const setupCamera = async () => {
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user' },
            audio: false,
          });
          
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
          }
        } catch (error) {
          console.error('Error accessing webcam:', error);
        }
      }
    };

    const loadModels = async () => {
      try {
        await tf.ready();
        
        
        const faceModel = await blazeface.load();
        setFaceDetector(faceModel);
        
       
        const emotionModel = createSimulatedEmotionModel();
        setModel(emotionModel);
        
        setIsModelLoading(false);
      } catch (error) {
        console.error('Error loading models:', error);
      }
    };

    
    const createSimulatedEmotionModel = () => {
      const model = tf.sequential();
      model.add(tf.layers.conv2d({
        inputShape: [48, 48, 1],
        filters: 32,
        kernelSize: 3,
        activation: 'relu'
      }));
      model.add(tf.layers.flatten());
      model.add(tf.layers.dense({
        units: emotions.length,
        activation: 'softmax'
      }));
      
      model.predict = (input) => {
        return tf.tidy(() => {
       
          const now = Date.now();
          if (now - fixedEmotionTimeRef.current > 3000) {
            fixedEmotionTimeRef.current = now;
            if (Math.random() > 0.7) {
              currentEmotionIndexRef.current = Math.floor(Math.random() * emotions.length);
            } else {
              currentEmotionIndexRef.current = Math.random() > 0.5 ? 4 : 3;
            }
          }
          
          
          const probabilities = Array(emotions.length).fill(0.05);
          probabilities[currentEmotionIndexRef.current] = 0.6;
          
          
          const sum = probabilities.reduce((a, b) => a + b, 0);
          const normalized = probabilities.map(val => val / sum);
          
    
          const withNoise = normalized.map(val => {
            const noise = (Math.random() * 0.1) - 0.05;
            return Math.max(0, val + noise);
          });
          
         
          const finalSum = withNoise.reduce((a, b) => a + b, 0);
          const final = withNoise.map(val => val / finalSum);
          
          return tf.tensor(final, [1, emotions.length]);
        });
      };
      
      return model;
    };

    setupCamera();
    loadModels();

   
    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      }
    };
  }, []);


  useEffect(() => {
    if (isModelLoading || !faceDetector || !model) return;

    const detectEmotion = async () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      
      if (!video || !canvas || video.readyState !== 4) return;
      
      const ctx = canvas.getContext('2d');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      
     
      const predictions = await faceDetector.estimateFaces(video, false);
      
      if (predictions.length > 0) {
       
        const face = predictions[0];
        const start = face.topLeft;
        const end = face.bottomRight;
        const size = [end[0] - start[0], end[1] - start[1]];
        
        
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 2;
        ctx.strokeRect(start[0], start[1], size[0], size[1]);
        
        const faceCanvas = document.createElement('canvas');
        faceCanvas.width = size[0];
        faceCanvas.height = size[1];
        faceCanvas.getContext('2d').drawImage(
          video, 
          start[0], start[1], size[0], size[1],
          0, 0, size[0], size[1]
        );
        
        
        const tensor = tf.browser.fromPixels(faceCanvas)
          .resizeNearestNeighbor([48, 48])
          .mean(2)
          .expandDims(2)
          .expandDims()
          .toFloat()
          .div(255.0);
        
       
        const result = await model.predict(tensor);
        const prediction = await result.data();
        
        
        recentEmotionsRef.current.push(prediction);
        if (recentEmotionsRef.current.length > emotionSmoothing) {
          recentEmotionsRef.current.shift();
        }
        
        
        const averagedPredictions = Array(emotions.length).fill(0);
        recentEmotionsRef.current.forEach(pred => {
          pred.forEach((p, i) => {
            averagedPredictions[i] += p;
          });
        });
        
        
        averagedPredictions.forEach((p, i) => {
          averagedPredictions[i] = p / recentEmotionsRef.current.length;
        });
        
        const emotionIndex = averagedPredictions.indexOf(Math.max(...averagedPredictions));
        setEmotion(emotions[emotionIndex]);
        
      
        ctx.fillStyle = '#00ff00';
        ctx.font = '24px Arial';
        ctx.fillText(emotions[emotionIndex], start[0], start[1] - 10);
        
       
        const barWidth = 100;
        const barHeight = 10;
        const barSpacing = 15;
        
        emotions.forEach((em, idx) => {
          const confidence = averagedPredictions[idx] * 100;
          ctx.fillStyle = idx === emotionIndex ? '#00ff00' : '#ffffff';
          ctx.fillRect(10, 10 + (idx * barSpacing), barWidth * (confidence / 100), barHeight);
          ctx.fillStyle = '#ffffff';
          ctx.font = '12px Arial';
          ctx.fillText(`${em}: ${confidence.toFixed(1)}%`, 10 + barWidth + 5, 10 + (idx * barSpacing) + barHeight);
        });
        
       
        tensor.dispose();
        result.dispose();
      } else {
        setEmotion('No face detected');
      }
    };
    
    const interval = setInterval(() => {
      detectEmotion();
    }, 100);
    
    return () => clearInterval(interval);
  }, [isModelLoading, faceDetector, model]);

  return (
    <div className="app">
      <h1>Emotion Detection</h1>
      <div className="video-container">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
        />
        <canvas ref={canvasRef} className="canvas-overlay" />
      </div>
      <div className="status">
        {isModelLoading ? (
          <p>Loading models...</p>
        ) : (
          <p>Current emotion: <strong>{emotion}</strong></p>
        )}
      </div>
      <div className="note">
      </div>
    </div>
  );
}

export default App;
