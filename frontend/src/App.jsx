
import './App.css'
import {useState} from 'react'

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [result, setResult] = useState(null)

  async function handleFileUpload() {
    if (!selectedFile) return 
    const formData = new FormData()
    formData.append("file", selectedFile)
    const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      })
      const data = await response.json()
      setResult(data)
    }
  
  
  return (
    <div>
      <h1> Deepfake Voice Detector </h1>
      <input 
        type="file" 
        accept="audio/*" 
        onChange={(e) => setSelectedFile(e.target.files[0])}
      />
      {selectedFile && <p>Picked: {selectedFile.name}</p>}
      {result && <p>Result: {result.prediction}</p>}
      <button onClick={handleFileUpload}>Check Audio</button>
    </div>
  )
 
}
export default App
