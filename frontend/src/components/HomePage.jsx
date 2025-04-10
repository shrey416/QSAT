// frontend/src/pages/HomePage.jsx
import React, { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import { v4 as uuidv4 } from 'uuid';

// --- Constants ---
const API_BASE_URL = 'http://localhost:5000/api'; // Backend URL

const ALL_WAVELENGTHS = [
  '410', '435', '460', '485', '510', '535', '560', '585',
  '610', '645', '680', '705', '730', '760', '810', '860',
  '900', '940'
];
const MIN_WAVELENGTH_INPUTS = 3;
const MAX_WAVELENGTH_INPUTS = ALL_WAVELENGTHS.length;
const MIN_TOP_X = 3;
const MAX_TOP_X = ALL_WAVELENGTHS.length;

// Display names for the UI
const METRIC_PARAMETERS = [
    'Capacity Moist', 'Temperature', 'Moisture', 'Electrical Conductivity', 'pH', 'Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)'
];

// Mapping from UI display names to frontend internal keys (used for TopX selection, analysis results display)
const METRIC_PARAM_KEYS_FRONTEND = {
    'Capacity Moist': 'capacityMoist', 'Temperature': 'temperature',
    'Moisture': 'moisture', 'Electrical Conductivity': 'electricalConductivity',
    'pH': 'pH', 'Nitrogen (N)': 'nitro',
    'Phosphorus (P)': 'phosphorus', 'Potassium (K)': 'potassium'
};

// Mapping from UI display names to *backend* keys (used for accessing metrics data)
const METRIC_PARAM_KEYS_BACKEND = {
    'Capacity Moist': 'Capacitity Moist', // Note the typo from your model
    'Temperature': 'Temp',
    'Moisture': 'Moist',
    'Electrical Conductivity': 'EC',
    'pH': 'Ph', // Note the capitalization from your model
    'Nitrogen (N)': 'Nitro',
    'Phosphorus (P)': 'Posh Nitro',
    'Potassium (K)': 'Pota Nitro'
};


const SIMULATED_WATER_LEVELS_KEYS = ['0ml', '25ml', '50ml'];

// --- Static Correct Wavelength Ranking Data ---
// Based on the provided table, using FRONTEND keys for easier lookup
const CORRECT_WAVELENGTH_RANKS = {
    // Frontend Key: { Wavelength: Rank }
    pH: { '410': 1, '435': 2, '460': 7, '485': 14, '510': 4, '535': 12, '560': 6, '585': 5, '610': 8, '645': 3, '680': 9, '705': 17, '730': 15, '760': 13, '810': 16, '860': 10, '900': 11, '940': 18 },
    nitro: { '410': 1, '435': 3, '460': 4, '485': 16, '510': 6, '535': 13, '560': 5, '585': 8, '610': 12, '645': 2, '680': 7, '705': 14, '730': 17, '760': 15, '810': 10, '860': 9, '900': 11, '940': 18 },
    phosphorus: { '410': 2, '435': 9, '460': 5, '485': 16, '510': 3, '535': 13, '560': 4, '585': 14, '610': 12, '645': 1, '680': 8, '705': 15, '730': 17, '760': 10, '810': 11, '860': 7, '900': 6, '940': 18 }, // Mapped from Posh Nitro
    potassium: { '410': 2, '435': 7, '460': 4, '485': 16, '510': 3, '535': 10, '560': 5, '585': 8, '610': 12, '645': 1, '680': 6, '705': 17, '730': 15, '760': 14, '810': 13, '860': 11, '900': 9, '940': 18 }, // Mapped from Pota Nitro
    capacityMoist: { '410': 2, '435': 9, '460': 1, '485': 15, '510': 7, '535': 11, '560': 10, '585': 12, '610': 14, '645': 8, '680': 6, '705': 17, '730': 16, '760': 13, '810': 5, '860': 3, '900': 4, '940': 18 }, // Mapped from Capacitity Moist
    temperature: { '410': 6, '435': 2, '460': 7, '485': 16, '510': 4, '535': 9, '560': 12, '585': 13, '610': 11, '645': 3, '680': 5, '705': 15, '730': 17, '760': 14, '810': 8, '860': 10, '900': 1, '940': 18 }, // Mapped from Temp
    moisture: { '410': 2, '435': 8, '460': 5, '485': 16, '510': 3, '535': 14, '560': 10, '585': 13, '610': 9, '645': 1, '680': 6, '705': 15, '730': 17, '760': 12, '810': 11, '860': 7, '900': 4, '940': 18 }, // Mapped from Moist
    electricalConductivity: { '410': 2, '435': 11, '460': 3, '485': 16, '510': 4, '535': 9, '560': 5, '585': 10, '610': 7, '645': 1, '680': 6, '705': 15, '730': 17, '760': 12, '810': 11, '860': 7, '900': 4, '940': 18 } // Mapped from EC
};


// --- Gemini Chat Constants ---
const DEFAULT_SYSTEM_PROMPT = `You are a helpful Soil Health Assistant. Your goal is to analyze the provided soil data and offer clear, concise, and actionable recommendations for improving soil health, fertility, and potentially crop yield based on the given parameters. Focus on practical advice.

You will be provided with the current soil analysis results preceding each user question. ALWAYS base your answers and recommendations strictly on this provided data. Disregard climatic and regional conditions unless explicitly mentioned in the user's follow-up.

Important Notes for Interpretation:
- At water level 0 NPK values are going to be very close to 0.
- The unit of NPK (Nitrogen, Phosphorus, Potassium) is mg/10gm, NOT kg/ha or other common units. Infer NPK levels based on this specific unit.
- Capacity Moist unit is %.
- Temperature unit is °C.
- Moisture value is a relative reading (often %).
- Electrical Conductivity (EC) unit is u/10gm (micro siemens per 10 grams).
- pH is a standard scale value.

Crop Recommendations: When asked for crop recommendations, base your suggestions *only* on the provided soil analysis results. Do not consider climate or region.

Response Format: Generate all text in plain text form without any bold, italics, or underlining.

Scope Limitations: Only respond to requests related to soil health, farming, crops, plants, trees, soil analysis interpretation, and farmer assistance based on the provided data. For any request outside this scope, respond *only* with: "Sorry I am not able to help you with that"`;


const USER_ROLE = 'user';
const MODEL_ROLE = 'model'; // To match Gemini's role naming

function HomePage() {
  // --- State ---
  const [waterLevel, setWaterLevel] = useState('');
  const [numWavelengthInputs, setNumWavelengthInputs] = useState(MIN_WAVELENGTH_INPUTS);
  // *** NEW: State for accuracy target selection ***
  const [accuracyTarget, setAccuracyTarget] = useState('general'); // 'general' or a metric key like 'pH'
  const [dynamicWavelengthData, setDynamicWavelengthData] = useState(
    () => initializeWavelengthRows(MIN_WAVELENGTH_INPUTS)
  );
  const [submitted, setSubmitted] = useState(false);
  const [analysisData, setAnalysisData] = useState(null);
  const [analysisError, setAnalysisError] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [metricsData, setMetricsData] = useState(null);
  const [metricsError, setMetricsError] = useState(null);
  const [isLoadingMetrics, setIsLoadingMetrics] = useState(true);
  const [topXCount, setTopXCount] = useState(MIN_TOP_X);
  const [selectedTopXAttribute, setSelectedTopXAttribute] = useState(METRIC_PARAM_KEYS_FRONTEND['pH']);
  const [rankedWavelengthsData, setRankedWavelengthsData] = useState([]); // State to hold the processed ranking data for display
  const [topXError, setTopXError] = useState(null); // Error state specifically for the ranking table
  const [isLoadingTopX, setIsLoadingTopX] = useState(false); // Loading state specifically for the ranking table
  const [currentView, setCurrentView] = useState('form');
  const [selectedMetricType, setSelectedMetricType] = useState('MAE');
  // --- Gemini Chat State ---
  const [showChatCard, setShowChatCard] = useState(false);
  const [chatHistory, setChatHistory] = useState([]); // Array of { role: 'user'/'model', text: 'message' }
  const [userInput, setUserInput] = useState('');
  const [isGeneratingInsights, setIsGeneratingInsights] = useState(false);
  const [geminiError, setGeminiError] = useState(null);


  // --- Refs ---
  const outputRef = useRef(null); // For scrolling to analysis results
  const chatCardRef = useRef(null); // For scrolling to chat card
  const chatHistoryRef = useRef(null); // For scrolling chat history

  // --- Helper Functions ---
  function initializeWavelengthRows(count) {
    return Array.from({ length: count }, () => ({
      id: uuidv4(), wavelength: '', value: '',
    }));
  }

  const formatAnalysisDataForPrompt = useCallback((data) => {
      if (!data) return 'No analysis data available.';
      const unitMap = { capacityMoist: '%', temperature: '°C', moisture: '', electricalConductivity: 'u/10gm', pH: '', nitro: 'mg/10gm', phosphorus: 'mg/10gm', potassium: 'mg/10gm' };
      return METRIC_PARAMETERS.map(paramDisplayName => {
          const frontendKey = METRIC_PARAM_KEYS_FRONTEND[paramDisplayName];
          const value = data[frontendKey];
          const unit = unitMap[frontendKey] || '';
          let displayValue = (value === null || value === undefined) ? "N/A" : (typeof value === 'number' ? (Number.isInteger(value) ? value.toString() : value.toFixed(3)) : String(value));
          return `- ${paramDisplayName}: ${displayValue}${unit ? ` ${unit}` : ''}`;
      }).join('\n');
  }, []); // Depends only on constants


  // --- Effects ---
  // Fetch Metrics on Mount
  useEffect(() => {
    const fetchMetrics = async () => {
      setIsLoadingMetrics(true);
      setMetricsError(null);
      try {
        const response = await fetch(`${API_BASE_URL}/metrics`);
        if (!response.ok) { const errorData = await response.json().catch(() => ({})); throw new Error(errorData?.error || `HTTP error ${response.status}`); }
        const data = await response.json();
        setMetricsData(data);
      } catch (err) {
        console.error("Failed to load metrics:", err);
        setMetricsError(`Could not load metrics: ${err.message}. Check backend connection.`);
        setMetricsData(null);
      } finally {
        setIsLoadingMetrics(false);
      }
    };
    fetchMetrics();
  }, []);

  // *** MODIFIED: Effect to handle wavelength row changes based on numWavelengthInputs AND accuracyTarget ***
  useEffect(() => {
      const desiredCount = parseInt(numWavelengthInputs, 10);
      if (isNaN(desiredCount) || desiredCount < MIN_WAVELENGTH_INPUTS || desiredCount > MAX_WAVELENGTH_INPUTS) {
          // If count is invalid, don't change anything yet, blur handler will fix it.
          return;
      }

      // --- Case 1: Specific Accuracy Target is selected ---
      if (accuracyTarget !== 'general') {
          const ranksForTarget = CORRECT_WAVELENGTH_RANKS[accuracyTarget];
          if (!ranksForTarget) {
              console.error(`No ranking data found for target: ${accuracyTarget}`);
              // Fallback to general behavior if ranks are missing? Or show error?
              // For now, let's clear the rows or revert to general? Let's keep general behavior for safety.
              setAccuracyTarget('general'); // Revert if data is bad
              return;
          }

          // Get top N ranked wavelengths for the target
          const sortedWavelengths = Object.entries(ranksForTarget)
              .map(([wavelength, rank]) => ({ wavelength, rank }))
              .sort((a, b) => a.rank - b.rank)
              .slice(0, desiredCount) // Get only the top N
              .map(item => item.wavelength);

          // Create new rows with locked wavelengths and cleared values
          const newRows = sortedWavelengths.map(wl => ({
              id: uuidv4(),
              wavelength: wl,
              value: '', // Clear value when target/count changes
          }));

          // If the number of rows is less than desiredCount (e.g., first time setting target), fill remaining
           while (newRows.length < desiredCount && newRows.length < ALL_WAVELENGTHS.length) {
              // This part should ideally not be reached if sortedWavelengths has enough items,
              // but as a fallback, add empty rows (though they won't have locked wavelengths)
               console.warn("Needed more rows than available top wavelengths for target. Adding empty rows.");
               newRows.push({ id: uuidv4(), wavelength: '', value: '' });
           }

          setDynamicWavelengthData(newRows);

      // --- Case 2: General Accuracy Target (User selects wavelengths) ---
      } else {
           // Only adjust row count if it doesn't match desired count
          if (desiredCount !== dynamicWavelengthData.length) {
              if (desiredCount > dynamicWavelengthData.length) {
                  // Add new empty rows
                  const newRowsToAdd = Array.from({ length: desiredCount - dynamicWavelengthData.length }, () => ({ id: uuidv4(), wavelength: '', value: '' }));
                  setDynamicWavelengthData(prevData => [...prevData, ...newRowsToAdd]);
              } else {
                  // Remove rows from the end
                  setDynamicWavelengthData(prevData => prevData.slice(0, desiredCount));
              }
          }
          // If count matches, but target changed TO 'general', ensure wavelengths are editable (they are by default)
          // No explicit action needed here as the disabling logic is in the render function.
      }

  }, [numWavelengthInputs, accuracyTarget]); // Rerun when count or target changes

  // Effect to process and set static Top Wavelengths for the metrics view
  useEffect(() => {
    if (!selectedTopXAttribute) return;
    setIsLoadingTopX(true);
    setTopXError(null);
    setRankedWavelengthsData([]);
    try {
        const ranksForAttribute = CORRECT_WAVELENGTH_RANKS[selectedTopXAttribute];
        if (!ranksForAttribute) { throw new Error(`Ranking data unavailable for the selected attribute.`); }
        const allRankedItems = Object.entries(ranksForAttribute)
            .map(([wavelength, rank]) => ({ wavelength: wavelength, rank: rank, importanceScore: null }))
            .sort((a, b) => a.rank - b.rank);
        const topXItems = allRankedItems.slice(0, topXCount);
        setRankedWavelengthsData(topXItems);
    } catch (err) {
        console.error("Error processing static ranking data:", err);
        setTopXError(`Could not load rankings: ${err.message}.`);
        setRankedWavelengthsData([]);
    } finally {
        setIsLoadingTopX(false);
    }
  }, [selectedTopXAttribute, topXCount]); // Re-run when attribute or count changes


  // Scroll Chat History to Bottom
  useEffect(() => {
    if (chatHistoryRef.current) {
      chatHistoryRef.current.scrollTop = chatHistoryRef.current.scrollHeight;
    }
  }, [chatHistory]);

  // --- Event Handlers ---
  const handleNumWavelengthChange = (e) => { setNumWavelengthInputs(e.target.value); };
  const handleNumWavelengthBlur = (e) => { let count = parseInt(e.target.value, 10); if (isNaN(count) || count < MIN_WAVELENGTH_INPUTS) { setNumWavelengthInputs(MIN_WAVELENGTH_INPUTS); } else if (count > MAX_WAVELENGTH_INPUTS) { setNumWavelengthInputs(MAX_WAVELENGTH_INPUTS); } else { setNumWavelengthInputs(count); } };
  // *** NEW: Handler for accuracy target change ***
  const handleAccuracyTargetChange = (e) => {
      const newTarget = e.target.value;
      setAccuracyTarget(newTarget);
      // Wavelength updates are handled by the useEffect hook reacting to accuracyTarget change
  };
  // *** MODIFIED: Wavelength change handler respects accuracyTarget ***
  const handleDynamicWavelengthChange = (id, selectedWavelength) => {
      // Only allow change if accuracy target is 'general'
      if (accuracyTarget === 'general') {
         setDynamicWavelengthData(prev => prev.map(row => (row.id === id ? { ...row, wavelength: selectedWavelength } : row)));
      }
      // If a specific target is selected, the dropdown is disabled, so this shouldn't be called.
  };
  const handleDynamicValueChange = (id, newValue) => { if (newValue === '' || /^-?\d*\.?\d*$/.test(newValue)) { setDynamicWavelengthData(prev => prev.map(row => (row.id === id ? { ...row, value: newValue } : row))); } };
  const handleTopXCountChange = (e) => { setTopXCount(e.target.value); };
  const handleTopXCountBlur = (e) => { let count = parseInt(e.target.value, 10); if (isNaN(count) || count < MIN_TOP_X) { setTopXCount(MIN_TOP_X); } else if (count > MAX_TOP_X) { setTopXCount(MAX_TOP_X); } };

  // Memoize selected wavelengths (only relevant when accuracyTarget is 'general')
  const getSelectedWavelengths = useMemo(() => {
    // If a target is selected, all wavelengths are predetermined, so uniqueness check isn't needed in the same way.
    // If general, we need to track user selections.
    if (accuracyTarget === 'general') {
        return dynamicWavelengthData.map(row => row.wavelength).filter(Boolean);
    }
    return []; // Return empty array if target is selected, as selection is locked
  }, [dynamicWavelengthData, accuracyTarget]);

  // Handle Soil Analysis Submission
  const handleSubmit = async (event) => {
    event.preventDefault();
    setAnalysisError(null);
    setIsAnalyzing(true);
    setSubmitted(false);
    setAnalysisData(null);
    setShowChatCard(false);
    setChatHistory([]);
    setGeminiError(null);

    let isValid = true;
    let errorMsg = '';
    const parsedWaterLevel = parseFloat(waterLevel);
    if (isNaN(parsedWaterLevel)) { errorMsg = 'Please enter a valid numerical water level.'; isValid = false; }

    // Validation for wavelength/value pairs
    const wavelengthPayload = {};
    const selectedWls = new Set();
    let filledRowCount = 0;

    if (isValid) {
        for (const row of dynamicWavelengthData) {
            // A row is considered "filled" if it has both a wavelength and a non-empty value
             if (row.wavelength && row.value.trim() !== '') {
                filledRowCount++;
                const parsedValue = parseFloat(row.value);
                if (isNaN(parsedValue)) { errorMsg = `Invalid value entered for wavelength ${row.wavelength}. Please enter a number.`; isValid = false; break; }
                // Duplicate check (more relevant for 'general' mode, but doesn't hurt)
                if (selectedWls.has(row.wavelength)) { errorMsg = `Duplicate wavelength detected: ${row.wavelength}. Wavelengths must be unique.`; isValid = false; break; }
                wavelengthPayload[row.wavelength] = parsedValue;
                selectedWls.add(row.wavelength);
            } else if (row.wavelength && row.value.trim() === '') {
                 // If wavelength is selected/locked but value is empty
                 errorMsg = `Please enter a value for wavelength ${row.wavelength}.`; isValid = false; break;
            } else if (accuracyTarget === 'general' && !row.wavelength && row.value.trim() !== '') {
                 // If value entered but no wavelength selected (only in general mode)
                 errorMsg = `Please select a wavelength for the entered value '${row.value}'.`; isValid = false; break;
            }
            // If accuracyTarget is set, row.wavelength should always be present due to useEffect.
            // If row.wavelength is empty in 'general' mode with empty value, it's just an unused row - skip.
        }
    }

    // Check if minimum number of *valid* rows is met
    if (isValid && filledRowCount < MIN_WAVELENGTH_INPUTS) {
        errorMsg = `Please provide valid absorbance values for at least ${MIN_WAVELENGTH_INPUTS} wavelengths. You have provided ${filledRowCount}.`;
        isValid = false;
    }

    if (!isValid) { setAnalysisError(errorMsg); setIsAnalyzing(false); return; }

    const payload = { waterLevel: parsedWaterLevel, wavelengths: wavelengthPayload };
    console.log("Sending analysis payload:", JSON.stringify(payload));

    try {
      const response = await fetch(`${API_BASE_URL}/analyze`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      const responseBody = await response.text();

      if (!response.ok) {
         let apiErrorMsg = `Analysis request failed: ${response.status}`;
         try { const errorData = JSON.parse(responseBody); apiErrorMsg = errorData?.error || apiErrorMsg; } catch (e) { /* Ignore JSON parse error on error response */ }
         throw new Error(apiErrorMsg);
      }
      let receivedData;
      try { receivedData = JSON.parse(responseBody); } catch (e) { throw new Error("Received invalid JSON response from server."); }
      if (!receivedData || typeof receivedData !== 'object') { throw new Error("Received unexpected data format from server."); }

      setAnalysisData(receivedData);
      setSubmitted(true);
      setAnalysisError(null);
      setCurrentView('form');

      setTimeout(() => { outputRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }); }, 150);

    } catch (error) {
      console.error("Error submitting analysis:", error);
      setAnalysisError(`Analysis failed: ${error.message}.`);
      setSubmitted(false);
      setAnalysisData(null);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // View Switching Handlers
  const handleViewMetrics = () => { if (!isLoadingMetrics) { setCurrentView('metrics'); window.scrollTo(0, 0); } };
  const handleBackToForm = () => {
      setCurrentView('form');
       if (submitted && analysisData) {
           // Scroll to results if they exist
           setTimeout(() => { outputRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }); }, 100);
       } else {
            // Scroll to top if no results yet
           window.scrollTo(0, 0);
       }
  };

  // --- Gemini Chat Handlers ---
  const callGeminiAPI = useCallback(async (promptToSend) => {
    setIsGeneratingInsights(true);
    setGeminiError(null);
    console.log("Calling Gemini API. Prompt length:", promptToSend.length);

    try {
        const response = await fetch(`${API_BASE_URL}/get-insights`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message: promptToSend }) });
        const responseBody = await response.text();

        if (!response.ok) {
            let errorMsg = `Gemini request failed: ${response.status}`;
            try { const errorData = JSON.parse(responseBody); errorMsg = errorData?.error || errorMsg; } catch (e) { /* Ignore */ }
            console.error("Gemini API Error Response Body:", responseBody);
            throw new Error(errorMsg);
        }
        let data;
         try { data = JSON.parse(responseBody); } catch (e) { console.error("Invalid JSON received from Gemini endpoint:", responseBody); throw new Error("Received invalid JSON response from Gemini endpoint."); }
        if (data && data.response) {
            setChatHistory(prev => [...prev, { role: MODEL_ROLE, text: data.response }]);
        } else { throw new Error("Received empty response from Gemini."); }
    } catch (error) {
        console.error("Error calling Gemini API:", error);
        setGeminiError(`Failed to get insights: ${error.message}`);
    } finally {
        setIsGeneratingInsights(false);
    }
  }, []); // No direct dependencies needed here


  const handleGetInsightsClick = useCallback(() => {
    if (!analysisData) { setGeminiError("Please run an analysis first to get insights."); return; }
    setShowChatCard(true);
    setGeminiError(null);
    setIsGeneratingInsights(true);

    const formattedData = formatAnalysisDataForPrompt(analysisData);
    const initialUserRequest = "Provide initial insights and crop recommendations based on the provided soil data.";
    const fullInitialPrompt = `${DEFAULT_SYSTEM_PROMPT}\n\n--- Current Soil Analysis Data ---\n${formattedData}\n\n--- User Request ---\nProvide initial insights and crop recommendations based *only* on the provided soil data. Disregard climate/region\ngive the recommendation of crops in the form of a list\n1. crop\nwhy this crop is suitable for the soil\n2. crop etc....`;

    setChatHistory([{ role: USER_ROLE, text: initialUserRequest }]);

    // *** MODIFIED: Scroll after API call potentially finishes/updates state ***
    callGeminiAPI(fullInitialPrompt).finally(() => {
         // Use finally to ensure scroll happens even if API fails, but state update might take time
         // Added a slightly longer delay to allow rendering after state update
         setTimeout(() => {
             chatCardRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
         }, 250); // Increased delay slightly
    });
  }, [analysisData, callGeminiAPI, formatAnalysisDataForPrompt]);

  const handleUserInput = (e) => { setUserInput(e.target.value); };

  const handleSendMessage = useCallback((e) => {
     if (e) e.preventDefault();
    const trimmedInput = userInput.trim();
    if (!trimmedInput || isGeneratingInsights || !analysisData) {
       if (!analysisData) { setGeminiError("Cannot send message: Soil analysis data context is missing."); }
      return;
    }
    setChatHistory(prev => [...prev, { role: USER_ROLE, text: trimmedInput }]);
    setUserInput('');
    const formattedData = formatAnalysisDataForPrompt(analysisData);
    const fullPrompt = `${DEFAULT_SYSTEM_PROMPT}\n\n--- Current Soil Analysis Data ---\n${formattedData}\n\n--- User Question ---\n${trimmedInput}`;
    callGeminiAPI(fullPrompt);
  }, [userInput, isGeneratingInsights, analysisData, callGeminiAPI, formatAnalysisDataForPrompt]);


  // --- Inline CSS ---
  const hideNumberInputArrows = `input[type='number']::-webkit-outer-spin-button, input[type='number']::-webkit-inner-spin-button {-webkit-appearance: none; margin: 0;} input[type='number'] {-moz-appearance: textfield;}`;
  const keyframes = `
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
    @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
    @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1.0); } }
  `;
  const animations = `
    .animate-fade-in { animation: fadeIn 0.5s ease-out forwards; }
    .animate-pulse-custom { animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
    .animate-spin-fast { animation: spin 1s linear infinite; }
    .animate-bounce { animation: bounce 1.4s infinite ease-in-out both; }
  `;


  // --- Render Logic ---

  const renderMetricsView = () => {
    if (isLoadingMetrics) return ( <div className="w-full max-w-6xl mx-auto mt-10 mb-20 text-center"><button onClick={handleBackToForm} className="mb-6 px-4 py-2 bg-blue-600 text-white rounded-lg shadow-md hover:bg-blue-700">← Back</button><div className="bg-white p-8 rounded-xl shadow-lg animate-pulse"><div className="h-8 bg-gray-200 rounded w-3/4 mx-auto mb-6"></div><div className="h-4 bg-gray-200 rounded w-1/2 mx-auto mb-8"></div><div className="space-y-4"><div className="h-10 bg-gray-200 rounded"></div><div className="h-10 bg-gray-200 rounded"></div><div className="h-10 bg-gray-200 rounded"></div></div></div></div> );
    if (metricsError || !metricsData) return ( <div className="w-full max-w-6xl mx-auto mt-10 mb-20 text-center"><button onClick={handleBackToForm} className="mb-6 px-4 py-2 bg-blue-600 text-white rounded-lg shadow-md hover:bg-blue-700">← Back</button><div className="bg-white p-8 rounded-xl shadow-lg border border-red-200"><h1 className="text-2xl font-bold text-red-600 mb-4">Error Loading Metrics</h1><p className="mt-4 text-gray-700">{metricsError || "Metrics data unavailable."}</p></div></div> );
    const getRankBgClass = (rank) => { if (rank === 1) return 'bg-yellow-100/60 hover:bg-yellow-200/70'; if (rank === 2) return 'bg-gray-200/60 hover:bg-gray-300/70'; if (rank === 3) return 'bg-orange-100/60 hover:bg-orange-200/70'; return 'bg-white hover:bg-gray-50'; };
    return (
        <div className="w-full max-w-6xl mx-auto mt-10 mb-20 space-y-8">
           <button onClick={handleBackToForm} className="mb-2 px-4 py-2 bg-blue-600 text-white font-semibold rounded-lg shadow-md hover:bg-blue-700 transition duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500">← Back to Analysis</button>
          {/* Main Metrics Card */}
          <div className="bg-white p-6 md:p-8 rounded-xl shadow-lg hover:shadow-xl transition-shadow duration-300">
            <h1 className="text-2xl md:text-3xl font-bold mb-6 text-gray-800">Model Performance Metrics</h1>
             <div className="mb-4 max-w-xs"> <label htmlFor="metric-type" className="block text-sm font-medium text-gray-700 mb-1">Select Metric:</label> <select id="metric-type" value={selectedMetricType} onChange={(e) => setSelectedMetricType(e.target.value)} className="block w-full px-3 py-2 border border-gray-300 bg-white rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 text-sm"> <option value="MAE">MAE (Lower is better)</option> <option value="RMSE">RMSE (Lower is better)</option> <option value="R2">R² Score (Higher is better)</option> </select> </div>
             <div className="mb-6 p-3 bg-blue-50 border border-blue-200 rounded-lg text-sm text-blue-700 shadow-sm"> <p><strong>MAE:</strong> Mean Absolute Error.</p> <p><strong>RMSE:</strong> Root Mean Squared Error.</p> <p><strong>R²:</strong> Coefficient of Determination.</p> </div>
            {/* *** MODIFIED: Added scrollbar styling classes *** */}
            <div className="overflow-x-auto rounded-lg border border-gray-200 shadow-sm scrollbar-thin scrollbar-thumb-gray-400 scrollbar-track-gray-100 pb-2">
                <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-100"><tr> <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider sticky left-0 bg-gray-100 z-10 whitespace-nowrap shadow-sm">Water Content</th> {METRIC_PARAMETERS.map(paramDisplayName => ( <th key={paramDisplayName} className="px-4 py-3 text-center text-xs font-semibold text-gray-600 uppercase tracking-wider whitespace-nowrap">{paramDisplayName}</th> ))} </tr></thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                        {SIMULATED_WATER_LEVELS_KEYS.map(levelKey => ( <tr key={levelKey} className="hover:bg-gray-50 transition-colors duration-150"> <td className="px-4 py-3 whitespace-nowrap text-sm font-medium text-gray-800 sticky left-0 bg-white hover:bg-gray-50 z-10 shadow-sm">{levelKey}</td>
                                {METRIC_PARAMETERS.map(paramDisplayName => { const backendKey = METRIC_PARAM_KEYS_BACKEND[paramDisplayName]; const value = metricsData?.[selectedMetricType]?.[levelKey]?.[backendKey]; let displayValue = 'N/A'; if (typeof value === 'number' && !isNaN(value)) { displayValue = value.toFixed(selectedMetricType === 'R2' ? 3 : 2); } else if (value !== undefined && value !== null) { displayValue = String(value); } return ( <td key={`${levelKey}-${backendKey}`} className="px-4 py-3 whitespace-nowrap text-sm text-gray-700 text-center"> <div className="transform transition duration-150 hover:scale-110 inline-block px-1">{displayValue}</div> </td> ); })}
                            </tr> ))}
                    </tbody>
                </table>
            </div> {/* End Metrics Table */}
          </div> {/* End Main Metrics Card */}

          {/* Top Wavelengths Card */}
          <div className="bg-white p-6 md:p-8 rounded-xl shadow-lg hover:shadow-xl transition-shadow duration-300">
             <h2 className="text-xl md:text-2xl font-bold mb-6 text-gray-800">Top Wavelength Importance</h2> <p className="text-sm text-gray-600 mb-6 -mt-4">Most influential wavelengths (based on pre-calculated ranks).</p>
            <div className="flex flex-col sm:flex-row gap-4 mb-6 items-end">
                <div className="flex-1 min-w-[200px]"> <label htmlFor="topx-attribute" className="block text-base font-medium text-gray-700 mb-2">Attribute:</label> <select id="topx-attribute" value={selectedTopXAttribute} onChange={(e) => setSelectedTopXAttribute(e.target.value)} className="block w-full px-4 py-3 border border-gray-300 bg-white rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 text-base transition duration-150"> {METRIC_PARAMETERS.map(paramDisplayName => ( <option key={METRIC_PARAM_KEYS_FRONTEND[paramDisplayName]} value={METRIC_PARAM_KEYS_FRONTEND[paramDisplayName]}> {paramDisplayName} </option> ))} </select> </div>
                 <div className="w-full sm:w-auto"> <label htmlFor="topx-count" className="block text-base font-medium text-gray-700 mb-2">Show Top:</label> <input id="topx-count" type="number" value={topXCount} onChange={handleTopXCountChange} onBlur={handleTopXCountBlur} min={MIN_TOP_X} max={MAX_TOP_X} required className="block w-full sm:w-28 px-4 py-3 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 text-base transition duration-150" placeholder={`(${MIN_TOP_X}-${MAX_TOP_X})`} /> </div>
            </div>
             {isLoadingTopX && ( <div className="text-center py-4 text-gray-500 animate-pulse-custom">Loading rankings...</div> )}
             {topXError && !isLoadingTopX && ( <div className="text-center py-4 text-red-600 bg-red-50 border border-red-200 rounded-md p-3">{topXError}</div> )}
             {!isLoadingTopX && !topXError && (
                <div className="overflow-x-auto rounded-lg border border-gray-200 shadow-sm max-w-md mx-auto">
                    <table className="min-w-full divide-y divide-gray-200">
                        <thead className="bg-gray-100"> <tr> <th className="px-6 py-3 text-center text-xs font-semibold text-gray-600 uppercase tracking-wider">Rank</th> <th className="px-6 py-3 text-center text-xs font-semibold text-gray-600 uppercase tracking-wider">Wavelength</th> </tr> </thead>
                        <tbody className="bg-white divide-y divide-gray-200">
                            {rankedWavelengthsData.length > 0 ? (
                              rankedWavelengthsData.map(item => (
                                <tr key={item.rank} className={`${getRankBgClass(item.rank)} transition-colors`}>
                                    <td className="px-6 py-3 text-sm font-medium text-gray-900 text-center"><span className={`inline-block px-2 py-0.5 rounded-full text-xs font-semibold ${item.rank <= 3 ? 'text-black' : 'text-gray-700'}`}>{item.rank}</span></td>
                                    <td className="px-6 py-3 text-sm text-gray-800 text-center">{item.wavelength}</td>
                                </tr>
                               ))
                             ) : (
                                <tr><td colSpan="2" className="text-center py-4 text-gray-500">No ranking data available.</td></tr> // Updated colspan
                             )}
                        </tbody>
                    </table>
                </div> )}
          </div> {/* End Top Wavelengths Card */}
        </div> // End Metrics View Container
      );
  };

  // Form and Output View
  const renderFormView = () => (
    <>
      {/* --- Input Card --- */}
      <div className="bg-white p-6 md:p-8 rounded-xl shadow-lg w-full max-w-4xl text-center mb-8 hover:shadow-xl transition-shadow duration-300">
        <h1 className="text-2xl md:text-3xl font-bold mb-4 md:mb-6 text-gray-800">Soil Spectrometer Analyzer</h1>
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Input Grid */}
          <div className="grid grid-cols-1 xl:grid-cols-2 xl:gap-x-8 gap-y-6">
            {/* Column 1: Water Level, Wavelength Count, Accuracy Target */}
            <div className="space-y-6">
              <div> <label htmlFor="water-level" className="block text-base font-medium text-gray-700 mb-2 text-left">Water Level (ml)</label> <input id="water-level" type="number" value={waterLevel} onChange={(e) => setWaterLevel(e.target.value)} required className="block w-full px-4 py-3 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 text-base transition duration-150" placeholder="e.g., 0, 25, 50" step="any" /> </div>
              <div> <label htmlFor="num-wavelengths" className="block text-base font-medium text-gray-700 mb-2 text-left">Wavelength Readings</label> <input id="num-wavelengths" type="number" value={numWavelengthInputs} onChange={handleNumWavelengthChange} onBlur={handleNumWavelengthBlur} min={MIN_WAVELENGTH_INPUTS} max={MAX_WAVELENGTH_INPUTS} required className="block w-full px-4 py-3 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 text-base transition duration-150" placeholder={`(${MIN_WAVELENGTH_INPUTS}-${MAX_WAVELENGTH_INPUTS})`} /> <p className="text-xs text-gray-500 mt-1 text-left">Number of wavelength inputs (Min {MIN_WAVELENGTH_INPUTS}).</p> </div>
              {/* *** NEW: Accuracy Target Selector *** */}
              <div>
                 <label htmlFor="accuracy-target" className="block text-base font-medium text-gray-700 mb-2 text-left">Maximise Accuracy For:</label>
                 <select
                    id="accuracy-target"
                    value={accuracyTarget}
                    onChange={handleAccuracyTargetChange}
                    className="block w-full px-4 py-3 border border-gray-300 bg-white rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 text-base transition duration-150"
                 >
                    <option value="general">General (Select Wavelengths)</option>
                    {METRIC_PARAMETERS.map(paramDisplayName => {
                        const frontendKey = METRIC_PARAM_KEYS_FRONTEND[paramDisplayName];
                        // Only include parameters for which we have rank data
                        if (CORRECT_WAVELENGTH_RANKS[frontendKey]) {
                            return (
                                <option key={frontendKey} value={frontendKey}>
                                    {paramDisplayName}
                                </option>
                            );
                        }
                        return null; // Skip if no rank data
                    })}
                 </select>
                 <p className="text-xs text-gray-500 mt-1 text-left">
                     {accuracyTarget === 'general'
                        ? 'Allows manual selection of wavelengths.'
                        : `Locks wavelength selection to the top ${numWavelengthInputs} ranked for ${Object.entries(METRIC_PARAM_KEYS_FRONTEND).find(([_, k]) => k === accuracyTarget)?.[0] ?? accuracyTarget}.`}
                </p>
              </div>
            </div>

            {/* Column 2: Dynamic Wavelength Inputs */}
            <div className="xl:col-span-1">
              <h2 className="text-lg md:text-xl font-semibold mb-3 text-left">Spectrometer Data (Absorbance)</h2>
              <div className="space-y-3 max-h-[400px] overflow-y-auto pr-2 -mr-2 border border-gray-200 rounded-md p-4 bg-gray-50/50 shadow-inner scrollbar-thin scrollbar-thumb-gray-300 scrollbar-track-gray-100">
                 {dynamicWavelengthData.map((row, index) => {
                     // Determine if wavelength selection should be disabled
                     const isWavelengthLocked = accuracyTarget !== 'general';
                     // Get available wavelengths only if in 'general' mode
                     const availableWavelengths = isWavelengthLocked ? [] : ALL_WAVELENGTHS.filter(wl => !getSelectedWavelengths.includes(wl) || wl === row.wavelength);

                     return (
                       <div key={row.id} className="grid grid-cols-2 gap-3 items-center">
                         <div>
                            <label htmlFor={`wavelength-select-${row.id}`} className="sr-only">Wavelength {index + 1}</label>
                            <select
                                id={`wavelength-select-${row.id}`}
                                value={row.wavelength}
                                onChange={(e) => handleDynamicWavelengthChange(row.id, e.target.value)}
                                // *** MODIFIED: Disable if accuracy target is selected ***
                                disabled={isWavelengthLocked}
                                required={!isWavelengthLocked} // Required only if user needs to select
                                className={`block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm transition duration-150 ${isWavelengthLocked ? 'bg-gray-100 text-gray-500 cursor-not-allowed' : 'bg-white'}`}
                            >
                               {/* Conditional rendering of options based on mode */}
                               {isWavelengthLocked ? (
                                   // If locked, just show the single, pre-filled wavelength
                                   <option value={row.wavelength}>{row.wavelength ? `${row.wavelength} nm` : 'N/A'}</option>
                                ) : (
                                   // If general mode, show standard options
                                   <>
                                      <option value="" disabled>Select Wavelength...</option>
                                      {/* Ensure current value is selectable even if duplicated elsewhere temporarily */}
                                      {row.wavelength && !availableWavelengths.includes(row.wavelength) && (
                                          <option key={row.wavelength} value={row.wavelength}>{row.wavelength} nm</option>
                                      )}
                                      {availableWavelengths.map(wl => (
                                          <option key={wl} value={wl}>{wl} nm</option>
                                      ))}
                                   </>
                               )}
                            </select>
                         </div>
                         <div>
                            <label htmlFor={`wavelength-value-${row.id}`} className="sr-only">Value for {row.wavelength || `Wavelength ${index + 1}`}</label>
                            <input
                               id={`wavelength-value-${row.id}`}
                               type="text"
                               inputMode="decimal"
                               value={row.value}
                               onChange={(e) => handleDynamicValueChange(row.id, e.target.value)}
                               required // Value is always required if the row is meant to be used
                               // Disable value input if wavelength is missing (can happen briefly or in error states)
                               disabled={!row.wavelength}
                               className={`block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm transition duration-150 ${!row.wavelength ? 'bg-gray-100 cursor-not-allowed opacity-70' : 'bg-white'}`}
                               placeholder="Absorbance"
                            />
                         </div>
                       </div>
                     );
                 })}
              </div>
              <p className="text-xs text-gray-500 mt-2 text-left">
                {accuracyTarget === 'general' ? '*Select unique wavelengths.' : '*Wavelengths locked by accuracy target.'}
              </p>
            </div>
          </div>

          {/* Error Display */}
          {analysisError && ( <div className="mt-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded-md text-sm text-left">{analysisError}</div> )}

          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 pt-2">
              <button type="submit" disabled={isAnalyzing || isLoadingMetrics} className={`w-full sm:w-auto flex-grow px-6 py-3 text-white font-semibold rounded-lg shadow-md transition duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 text-base transform hover:scale-105 ${isAnalyzing || isLoadingMetrics ? 'bg-gray-400 cursor-not-allowed' : 'bg-green-600 hover:bg-green-700'}`}> {isAnalyzing ? 'Analyzing...' : 'Analyze Soil Data'} </button>
              <button type="button" onClick={handleViewMetrics} disabled={isAnalyzing || isLoadingMetrics} className={`w-full sm:w-auto px-6 py-3 text-white font-semibold rounded-lg shadow-md transition duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 text-base transform hover:scale-105 ${ isAnalyzing || isLoadingMetrics ? 'bg-gray-400 cursor-not-allowed' : 'bg-indigo-600 hover:bg-indigo-700'}`} title={isLoadingMetrics ? "Metrics loading..." : "View performance metrics"}> {isLoadingMetrics ? 'Loading...' : 'View Performance'} </button>
          </div>
        </form>
      </div>

      {/* --- Output Section --- */}
      {submitted && analysisData && !isAnalyzing && (
        <div ref={outputRef} className="mt-8 w-full max-w-4xl space-y-8 md:max-w-6xl animate-fade-in">
           {/* Results Table */}
           <div className="bg-white p-6 rounded-xl shadow-lg hover:shadow-xl transition-shadow duration-300">
                <h2 className="text-xl md:text-2xl font-bold mb-4 text-gray-800">Soil Analysis Results</h2>
                <div className="overflow-x-auto rounded-lg border border-gray-200 shadow-sm">
                    <table className="min-w-full divide-y divide-gray-200">
                        <thead className="bg-gray-50"> <tr> <th className="px-4 py-3 text-left text-sm font-medium text-gray-500 uppercase tracking-wider">Attribute</th> <th className="px-4 py-3 text-left text-sm font-medium text-gray-500 uppercase tracking-wider">Predicted Value</th> </tr> </thead>
                        <tbody className="bg-white divide-y divide-gray-200">
                        {METRIC_PARAMETERS.map(paramDisplayName => { const frontendKey = METRIC_PARAM_KEYS_FRONTEND[paramDisplayName]; const value = analysisData[frontendKey]; const unitMap = { temperature: '°C', moisture: '', electricalConductivity: 'u/10gm', nitro: 'mg/10gm', phosphorus: 'mg/10gm', potassium: 'mg/10gm', capacityMoist: '%'}; const unit = unitMap[frontendKey] || ''; const displayValue = (value === null || value === undefined) ? "N/A" : (typeof value === 'number' ? value.toFixed(3) : value); return ( <tr key={frontendKey} className="hover:bg-gray-50 transition-colors"> <td className="px-4 py-3 whitespace-nowrap text-sm font-medium text-gray-700">{paramDisplayName}</td> <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900"> {displayValue !== 'N/A' ? `${displayValue}${unit ? ` ${unit}` : ''}` : 'N/A'} </td> </tr> ); })}
                        </tbody>
                    </table>
                </div>
            </div>

          {/* Key Indicators Dashboard */}
           <div className="bg-white p-6 rounded-xl shadow-lg hover:shadow-xl transition-shadow duration-300">
                <h2 className="text-xl md:text-2xl font-bold mb-4 text-gray-800">Key Indicators</h2>
                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4 mb-6">
                    {Object.entries(analysisData).filter(([key]) => ['pH', 'nitro', 'potassium', 'phosphorus', 'moisture', 'electricalConductivity', 'temperature', 'capacityMoist'].includes(key)).map(([frontendKey, value]) => { if (value === null || value === undefined) return null; const paramEntry = Object.entries(METRIC_PARAM_KEYS_FRONTEND).find(([name, k]) => k === frontendKey); const label = paramEntry ? paramEntry[0] : frontendKey; let unit = ''; let bgColor = 'bg-gray-50 border-gray-200'; let interpretation = ''; switch (frontendKey) { case 'pH': const phVal = parseFloat(value); if (!isNaN(phVal)) { bgColor = phVal < 6 ? 'bg-orange-50 border-orange-200' : phVal > 7.5 ? 'bg-blue-50 border-blue-200' : 'bg-green-50 border-green-200'; interpretation = phVal < 6 ? 'Acidic' : phVal > 7.5 ? 'Alkaline' : 'Neutral'; unit=''; } break; case 'nitro': unit = 'mg/10gm'; bgColor = 'bg-yellow-50 border-yellow-200'; break; case 'potassium': unit = 'mg/10gm'; bgColor = 'bg-red-50 border-red-200'; break; case 'phosphorus': unit = 'mg/10gm'; bgColor = 'bg-purple-50 border-purple-200'; break; case 'moisture': unit = ''; bgColor = 'bg-cyan-50 border-cyan-200'; break; case 'capacityMoist': unit = '%'; bgColor = 'bg-sky-50 border-sky-200'; break; case 'electricalConductivity': unit = 'u/10gm'; bgColor = 'bg-lime-50 border-lime-200'; break; case 'temperature': unit = '°C'; bgColor = 'bg-rose-50 border-rose-200'; break; default: break; } const displayValue = typeof value === 'number' ? value.toFixed(2) : value; return ( <div key={frontendKey} className={`p-4 rounded-lg shadow border transform transition duration-300 hover:scale-105 ${bgColor}`}> <h3 className="text-sm md:text-base font-semibold text-gray-700 truncate" title={label}>{label}</h3> <p className="text-lg md:text-xl font-bold text-gray-900">{displayValue}{unit ? <span className="text-sm font-normal"> {unit}</span> : ''}</p> {interpretation && <p className="text-xs text-gray-600">{interpretation}</p>} </div> ); })}
                </div>
                {/* --- Get Insights Button --- */}
                 <div className="mt-6 text-center">
                     <button
                         type="button"
                         onClick={handleGetInsightsClick} // Uses useCallback version
                         disabled={!analysisData || isGeneratingInsights || isAnalyzing}
                         className={`px-8 py-3 text-lg text-white font-semibold rounded-lg shadow-md transition duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-500 transform hover:scale-105 ${!analysisData || isGeneratingInsights || isAnalyzing ? 'bg-gray-400 cursor-not-allowed' : 'bg-purple-600 hover:bg-purple-700'}`}
                         title={!analysisData ? "Run analysis first" : "Get AI-powered insights"}
                     >
                         {isGeneratingInsights ? 'Loading Insights...' : 'Get AI Insights ✨'}
                     </button>
                 </div>
            </div>

             {/* --- Gemini Chat Card (conditionally rendered) --- */}
              {showChatCard && (
                <div ref={chatCardRef} className="mt-10 w-full max-w-4xl md:max-w-6xl animate-fade-in">
                    <div className="bg-gradient-to-br from-purple-50 via-indigo-50 to-blue-50 p-4 sm:p-6 rounded-xl shadow-lg border border-purple-200 hover:shadow-xl transition-shadow duration-300 flex flex-col h-[600px]"> {/* Fixed height */}
                         <h2 className="text-xl md:text-2xl font-bold mb-4 text-gray-800 text-center">Soil Health Assistant (Gemini)</h2>

                         {/* Chat History Area */}
                         <div ref={chatHistoryRef} className="flex-grow overflow-y-auto mb-4 space-y-4 pr-2 -mr-2 scrollbar-thin scrollbar-thumb-purple-300 scrollbar-track-purple-100">
                            {chatHistory.map((msg, index) => (
                                <div key={index} className={`flex ${msg.role === USER_ROLE ? 'justify-end' : 'justify-start'}`}>
                                    <div className={`max-w-[80%] p-3 rounded-xl shadow-md ${ msg.role === USER_ROLE ? 'bg-blue-500 text-white rounded-br-none' : msg.role === MODEL_ROLE ? 'bg-white text-gray-800 border border-gray-200 rounded-bl-none' : 'bg-yellow-100 text-yellow-800 text-sm italic border border-yellow-200' }`}>
                                       {msg.text.split('\n').map((line, i) => <p key={i} className="mb-1 last:mb-0">{line || <> </>}</p>)}
                                     </div>
                                </div>
                             ))}
                             {isGeneratingInsights && chatHistory.length > 0 && (
                                <div className="flex justify-start">
                                     <div className="p-3 rounded-lg bg-gray-200 text-gray-600 inline-flex items-center space-x-2 rounded-bl-none">
                                          <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                                          <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                                          <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></div>
                                     </div>
                                </div>
                            )}
                         </div>

                         {/* Error Display */}
                        {geminiError && ( <div className="mb-3 p-3 bg-red-100 border border-red-400 text-red-700 rounded-md text-sm text-left">{geminiError}</div> )}

                         {/* Input Area */}
                         <form onSubmit={handleSendMessage} className="flex items-center gap-3 pt-3 border-t border-purple-200">
                             <input
                                type="text"
                                value={userInput}
                                onChange={handleUserInput}
                                placeholder={isGeneratingInsights ? "Assistant is thinking..." : "Ask a follow-up question..."}
                                disabled={isGeneratingInsights || !analysisData}
                                className="flex-grow px-4 py-3 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-purple-500 text-base transition duration-150 disabled:bg-gray-100 disabled:cursor-not-allowed"
                                required
                             />
                             <button
                                type="submit"
                                disabled={isGeneratingInsights || !userInput.trim() || !analysisData}
                                className={`p-2.5 sm:p-3 text-white font-semibold rounded-lg shadow-md transition duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 ${isGeneratingInsights || !userInput.trim() || !analysisData ? 'bg-gray-400' : 'bg-blue-600 hover:bg-blue-700'}`}
                             >
                                <svg className='w-6 h-6 sm:w-7 sm:h-7' viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                  <path d="M11.5003 12H5.41872M5.24634 12.7972L4.24158 15.7986C3.69128 17.4424 3.41613 18.2643 3.61359 18.7704C3.78506 19.21 4.15335 19.5432 4.6078 19.6701C5.13111 19.8161 5.92151 19.4604 7.50231 18.7491L17.6367 14.1886C19.1797 13.4942 19.9512 13.1471 20.1896 12.6648C20.3968 12.2458 20.3968 11.7541 20.1896 11.3351C19.9512 10.8529 19.1797 10.5057 17.6367 9.81135L7.48483 5.24303C5.90879 4.53382 5.12078 4.17921 4.59799 4.32468C4.14397 4.45101 3.77572 4.78336 3.60365 5.22209C3.40551 5.72728 3.67772 6.54741 4.22215 8.18767L5.24829 11.2793C5.34179 11.561 5.38855 11.7019 5.407 11.8459C5.42338 11.9738 5.42321 12.1032 5.40651 12.231C5.38768 12.375 5.34057 12.5157 5.24634 12.7972Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                                </svg>
                             </button>
                         </form>
                    </div>
                 </div>
             )} {/* End Gemini Chat Card */}

        </div> // End Output Section Container
      )}
    </>
  );

  // Main Return
  return (
    <div className="min-h-screen flex flex-col items-center justify-start bg-gradient-to-br from-green-50 via-blue-50 to-purple-50 p-4 pt-10 pb-20">
       <style jsx="true" global="true">{`
        ${keyframes}
        ${animations}
        ${hideNumberInputArrows}
         th.sticky, td.sticky { position: sticky; left: 0; z-index: 10; }
         thead th.sticky { z-index: 20; box-shadow: 2px 0 5px -2px rgba(0,0,0,0.1); }
         tbody td.sticky { box-shadow: 2px 0 5px -2px rgba(0,0,0,0.1); }
         thead th.sticky:first-child { background-color: #f9fafb; } /* bg-gray-50 */
         tbody td.sticky:first-child { background-color: #ffffff; } /* bg-white */
         /* Ensure hover maintains sticky background */
         tr:hover td.sticky:first-child { background-color: #f9fafb; } /* bg-gray-50 on hover */
         /* Custom scrollbar general */
         .scrollbar-thin { scrollbar-width: thin; scrollbar-color: #9ca3af #f3f4f6; } /* thumb/track */
         .scrollbar-thin::-webkit-scrollbar { width: 8px; height: 8px; }
         .scrollbar-thin::-webkit-scrollbar-track { background: #f3f4f6; border-radius: 4px; }
         .scrollbar-thin::-webkit-scrollbar-thumb { background-color: #9ca3af; border-radius: 4px; border: 2px solid #f3f4f6; } /* border creates padding */
         .scrollbar-thin::-webkit-scrollbar-thumb:hover { background-color: #6b7280; }
         /* Custom scrollbar for specific elements */
         .scrollbar-thumb-gray-400::-webkit-scrollbar-thumb { background-color: #9ca3af; }
         .scrollbar-track-gray-100::-webkit-scrollbar-track { background: #f3f4f6; }
         .scrollbar-thumb-gray-400 { scrollbar-color: #9ca3af #f3f4f6; } /* For Firefox */
         .scrollbar-thumb-gray-300::-webkit-scrollbar-thumb { background-color: #d1d5db; }
         .scrollbar-track-gray-100::-webkit-scrollbar-track { background: #f3f4f6; }
         .scrollbar-thumb-gray-300 { scrollbar-color: #d1d5db #f3f4f6; } /* For Firefox */
         .scrollbar-thumb-purple-300::-webkit-scrollbar-thumb { background-color: #c4b5fd; }
         .scrollbar-track-purple-100::-webkit-scrollbar-track { background-color: #ede9fe; }
         .scrollbar-thumb-purple-300 { scrollbar-color: #c4b5fd #ede9fe; }
         /* Animations */
         .animate-bounce { animation: bounce 1.4s infinite ease-in-out both; }
         /* Ensure send button icon color is inherited */
         button svg path { stroke: currentColor; }
      `}</style>
      {currentView === 'form' ? renderFormView() : renderMetricsView()}
    </div>
  );
}

export default HomePage;