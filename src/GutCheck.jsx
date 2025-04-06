// GutCheck.jsx
import React, { useState, useEffect } from 'react';
import { Brain } from 'lucide-react';

const GutCheck = () => {
  const [stage, setStage] = useState('input'); // input, preparation, recording, processing, results
  const [question, setQuestion] = useState('');
  const [choices, setChoices] = useState([]);
  const [currentChoice, setCurrentChoice] = useState('');
  const [currentChoiceIndex, setCurrentChoiceIndex] = useState(0);
  const [countdownTime, setCountdownTime] = useState(10);
  const [isRecording, setIsRecording] = useState(false);
  const [results, setResults] = useState(null);
  const [isReadyForNext, setIsReadyForNext] = useState(false);

  // Handle adding a new choice
  const handleAddChoice = () => {
    if (currentChoice.trim() !== '') {
      setChoices([...choices, currentChoice.trim()]);
      setCurrentChoice('');
    }
  };

  // Handle submitting all choices
  const handleSubmitChoices = () => {
    if (choices.length < 2) {
      alert('Please enter at least two choices to compare.');
      return;
    }
    if (!question.trim()) {
      alert('Please enter your question.');
      return;
    }
    setStage('preparation');
  };

  // Handle starting the recording process
  const handleStartRecording = () => {
    setStage('recording');
    setCurrentChoiceIndex(0);
    setIsReadyForNext(false);
  };

  // Handle when user is ready for next choice
  const handleReadyForNext = () => {
    setIsReadyForNext(true);
    setCountdownTime(10);
    setIsRecording(true);
  };

  // Simulate recording completion
  useEffect(() => {
    let timer;
    if (isRecording && countdownTime > 0) {
      timer = setTimeout(() => setCountdownTime(countdownTime - 1), 1000);
    } else if (isRecording && countdownTime === 0) {
      setIsRecording(false);
      if (currentChoiceIndex < choices.length - 1) {
        setCurrentChoiceIndex(currentChoiceIndex + 1);
        setIsReadyForNext(false);
      } else {
        // All choices recorded, move to processing
        setStage('processing');
        // Simulate ML processing time
        setTimeout(() => {
          // Generate mock results (in a real app, these would come from the ML pipeline)
          const mockResults = choices.map((choice, index) => ({
            choice,
            confidence: Math.random() * 100
          })).sort((a, b) => b.confidence - a.confidence);
          
          setResults(mockResults);
          setStage('results');
        }, 5000);
      }
    }
    return () => clearTimeout(timer);
  }, [isRecording, countdownTime, currentChoiceIndex, choices.length]);

  // Render different stages of the application
  const renderStage = () => {
    switch (stage) {
      case 'input':
        return (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold mb-4">What's your question?</h2>
            <div>
              <input
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Enter your question here..."
                className="w-full p-3 border rounded mb-4"
              />
            </div>
            
            <h2 className="text-2xl font-bold mb-4">Enter Your Choices</h2>
            <div className="flex gap-2 mb-4">
              <input
                type="text"
                value={currentChoice}
                onChange={(e) => setCurrentChoice(e.target.value)}
                placeholder="Enter a possible answer..."
                className="flex-grow p-3 border rounded"
                onKeyPress={(e) => e.key === 'Enter' && handleAddChoice()}
              />
              <button 
                onClick={handleAddChoice}
                className="bg-blue-600 text-white px-4 py-2 rounded"
              >
                Add
              </button>
            </div>
            
            {choices.length > 0 && (
              <div className="mb-6">
                <h3 className="font-semibold mb-2">Your choices:</h3>
                <ul className="list-disc pl-5">
                  {choices.map((choice, index) => (
                    <li key={index} className="mb-1">{choice}</li>
                  ))}
                </ul>
              </div>
            )}
            
            <button
              onClick={handleSubmitChoices}
              disabled={choices.length < 2 || !question.trim()}
              className={`w-full p-3 rounded font-semibold ${
                choices.length < 2 || !question.trim() 
                  ? 'bg-gray-300 text-gray-600' 
                  : 'bg-green-600 text-white hover:bg-green-700'
              }`}
            >
              Continue to Recording
            </button>
          </div>
        );
        
      case 'preparation':
        return (
          <div className="text-center space-y-6">
            <h2 className="text-2xl font-bold mb-6">Prepare for Recording</h2>
            <p className="mb-6">
              We'll show you each choice one by one. For each choice:
            </p>
            <ol className="list-decimal text-left ml-6 mb-6 space-y-2">
              <li>Clear your mind before each choice</li>
              <li>Think about how you feel about the choice for 10 seconds</li>
              <li>The system will record your brain activity during this time</li>
            </ol>
            <p className="mb-6">
              Please make sure your EEG device is properly connected and working.
            </p>
            <button
              onClick={handleStartRecording}
              className="w-full p-3 bg-green-600 text-white rounded font-semibold hover:bg-green-700"
            >
              I'm Ready to Begin
            </button>
          </div>
        );
        
      case 'recording':
        return (
          <div className="text-center space-y-6">
            {!isReadyForNext ? (
              <>
                <h2 className="text-2xl font-bold mb-4">
                  Clear Your Mind
                </h2>
                <p className="mb-6">
                  Take a deep breath and try to clear your thoughts.
                </p>
                <p className="text-lg font-semibold mb-6">
                  When you're ready, we'll record your response to:
                </p>
                <div className="text-xl font-bold p-4 bg-blue-100 rounded-lg mb-6">
                  {choices[currentChoiceIndex]}
                </div>
                <button
                  onClick={handleReadyForNext}
                  className="w-full p-3 bg-blue-600 text-white rounded font-semibold hover:bg-blue-700"
                >
                  I'm Ready
                </button>
              </>
            ) : (
              <>
                <h2 className="text-2xl font-bold mb-4">
                  Think About This Choice
                </h2>
                <div className="text-xl font-bold p-4 bg-blue-100 rounded-lg mb-6">
                  {choices[currentChoiceIndex]}
                </div>
                <div className="text-center">
                  <div className="text-5xl font-bold mb-4">
                    {countdownTime}
                  </div>
                  <p className="text-lg">
                    {isRecording ? 'Recording in progress...' : 'Preparing next choice...'}
                  </p>
                </div>
              </>
            )}
            <div className="mt-4 text-sm text-gray-500">
              Choice {currentChoiceIndex + 1} of {choices.length}
            </div>
          </div>
        );
        
      case 'processing':
        return (
          <div className="text-center space-y-6">
            <h2 className="text-2xl font-bold mb-6">Processing Your Results</h2>
            <div className="flex justify-center mb-6">
              <div className="animate-pulse">
                <Brain size={100} className="text-blue-600" />
              </div>
            </div>
            <p className="text-lg">
              Our machine learning model is analyzing your brain activity...
            </p>
          </div>
        );
        
      case 'results':
        if (!results) return <div>No results available</div>;
        
        const bestChoice = results[0];
        const maxConfidence = Math.max(...results.map(r => r.confidence));
        
        return (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold mb-2">Your Results</h2>
            <p className="mb-6">Based on your brain activity, your gut is telling you:</p>
            
            <div className="bg-green-100 p-6 rounded-lg text-center mb-6">
              <h3 className="text-3xl font-bold text-green-800 mb-2">
                {bestChoice.choice}
              </h3>
              <p className="text-lg">
                Confidence: {bestChoice.confidence.toFixed(1)}%
              </p>
            </div>
            
            <details className="bg-gray-100 p-4 rounded-lg">
              <summary className="font-semibold cursor-pointer">
                See all choices comparison
              </summary>
              <div className="mt-4 space-y-4">
                {results.map((result, index) => (
                  <div key={index} className="mb-2">
                    <div className="flex justify-between mb-1">
                      <span>{result.choice}</span>
                      <span>{result.confidence.toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-gray-300 rounded-full h-2">
                      <div 
                        className="bg-blue-600 h-2 rounded-full" 
                        style={{ width: `${(result.confidence / maxConfidence) * 100}%` }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
            </details>
            
            <button
              onClick={() => {
                setStage('input');
                setChoices([]);
                setQuestion('');
                setResults(null);
              }}
              className="w-full p-3 bg-blue-600 text-white rounded font-semibold hover:bg-blue-700 mt-4"
            >
              Start Over
            </button>
          </div>
        );
        
      default:
        return <div>Unknown stage</div>;
    }
  };

  return (
    <div className="max-w-lg mx-auto p-6">
      <div className="flex items-center justify-center mb-8">
        <Brain className="text-blue-600 mr-2" size={32} />
        <h1 className="text-3xl font-bold">GutCheck</h1>
      </div>
      
      {renderStage()}
    </div>
  );
};

export default GutCheck;