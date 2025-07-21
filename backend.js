const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = 'uploads/';
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir);
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + '-' + file.originalname);
  }
});

const upload = multer({ storage: storage });

// Routes
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Upload files and run analysis
app.post('/api/analyze', upload.fields([
  { name: 'seatData', maxCount: 1 },
  { name: 'eventData', maxCount: 1 }
]), async (req, res) => {
  try {
    const { targetGame, targetDate } = req.body;
    
    if (!req.files.seatData || !req.files.eventData) {
      return res.status(400).json({ error: 'Both seat data and event data files are required' });
    }

    const seatDataPath = req.files.seatData[0].path;
    const eventDataPath = req.files.eventData[0].path;

    // Create a modified Python script with the uploaded file paths
    const pythonScript = createPythonScript(seatDataPath, eventDataPath, targetGame, targetDate);
    
    // Write the modified script to a temporary file
    const tempScriptPath = `temp_script_${Date.now()}.py`;
    fs.writeFileSync(tempScriptPath, pythonScript);

    // Run the Python script
    const pythonProcess = spawn('python', [tempScriptPath]);
    
    let outputData = '';
    let errorData = '';

    pythonProcess.stdout.on('data', (data) => {
      outputData += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      errorData += data.toString();
    });

    pythonProcess.on('close', (code) => {
      // Clean up temporary files
      fs.unlinkSync(tempScriptPath);
      fs.unlinkSync(seatDataPath);
      fs.unlinkSync(eventDataPath);

      if (code !== 0) {
        console.error('Python script error:', errorData);
        return res.status(500).json({ error: 'Analysis failed', details: errorData });
      }

      // Parse the output and send results
      try {
        const results = parseAnalysisOutput(outputData);
        res.json(results);
      } catch (parseError) {
        console.error('Parse error:', parseError);
        res.status(500).json({ error: 'Failed to parse analysis results', details: outputData });
      }
    });

  } catch (error) {
    console.error('Server error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get sample prediction
app.post('/api/predict', (req, res) => {
  const { daysUntilEvent, price, quantity, zone } = req.body;
  
  // This would normally call your trained model
  // For now, returning mock data based on your algorithm logic
  const optimalProb = calculateOptimalProbability(daysUntilEvent, price);
  const recommendation = getRecommendation(optimalProb);
  
  res.json({
    optimalProbability: optimalProb,
    predictedPrice: price * (0.95 + Math.random() * 0.1), // Mock price prediction
    recommendation: recommendation
  });
});

function createPythonScript(seatDataPath, eventDataPath, targetGame, targetDate) {
  // Read the original script and modify the file paths
  const originalScript = fs.readFileSync('ticket_optimizer.py', 'utf8');
  
  return originalScript
    .replace(/SEAT_DATA_PATH = ".*"/, `SEAT_DATA_PATH = "${seatDataPath.replace(/\\/g, '/')}"`)
    .replace(/EVENT_DATA_PATH = ".*"/, `EVENT_DATA_PATH = "${eventDataPath.replace(/\\/g, '/')}"`)
    .replace(/target_game = ".*"/, `target_game = "${targetGame}"`)
    .replace(/target_date = ".*"/, `target_date = "${targetDate}"`);
}

function parseAnalysisOutput(output) {
  // Parse the Python script output to extract key metrics
  const lines = output.split('\n');
  const results = {
    modelPerformance: {},
    timingAnalysis: {},
    featureImportance: [],
    recommendation: '',
    insights: []
  };

  let currentSection = '';
  
  for (let line of lines) {
    line = line.trim();
    
    if (line.includes('MODEL PERFORMANCE')) {
      currentSection = 'performance';
    } else if (line.includes('TIMING ANALYSIS')) {
      currentSection = 'timing';
    } else if (line.includes('FEATURE IMPORTANCE')) {
      currentSection = 'features';
    } else if (line.includes('RECOMMENDATION')) {
      currentSection = 'recommendation';
      results.recommendation = line.split('RECOMMENDATION: ')[1] || '';
    } else if (line.includes('SUMMARY INSIGHTS')) {
      currentSection = 'insights';
    }
    
    // Parse specific metrics
    if (line.includes('Classification Accuracy:')) {
      results.modelPerformance.accuracy = parseFloat(line.match(/[\d.]+/)[0]);
    } else if (line.includes('Regression R²:')) {
      results.modelPerformance.r2Score = parseFloat(line.match(/[\d.]+/)[0]);
    } else if (line.includes('Average optimal timing:')) {
      results.timingAnalysis.averageOptimalDays = parseFloat(line.match(/[\d.]+/)[0]);
    } else if (line.startsWith('•')) {
      results.insights.push(line.substring(1).trim());
    }
  }
  
  return results;
}

function calculateOptimalProbability(daysUntilEvent, price) {
  // Simple heuristic based on your model logic
  let prob = 0.5;
  
  if (daysUntilEvent >= 20 && daysUntilEvent <= 40) {
    prob += 0.3;
  }
  if (price > 100) {
    prob += 0.1;
  }
  
  return Math.min(0.95, Math.max(0.05, prob));
}

function getRecommendation(probability) {
  if (probability > 0.7) {
    return { status: 'buy', message: 'Good time to buy!', color: 'green' };
  } else if (probability > 0.4) {
    return { status: 'wait', message: 'Consider waiting a bit longer', color: 'orange' };
  } else {
    return { status: 'hold', message: 'Wait for better timing', color: 'red' };
  }
}

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});