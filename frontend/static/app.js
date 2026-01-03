//----------- FUNCTIONS ------------//

function handleIngestSuccess(data) {
    currentSessionId = data.session_id; // connects the current webpage to that document
    uploadStatus.textContent = "PDF processed successfully.";
    uploadStatus.style.color = "green";

    // Show summary section
    const summarySection = document.getElementById("summary-section");
    const summaryText = document.getElementById("summary-text");
    const documentMeta = document.getElementById("document-meta");

    summaryText.textContent = data.summary;

    documentMeta.textContent = `Pages: ${data.pages} || Characters: ${data.chars}`;

    // shows the summary section that was previously hidden
    summarySection.style.display = "block";

    // show the ask section after ingest success
    document.getElementById("ask-section").style.display = "block";
}


function handleAskSuccess(data) {
    askStatus.textContent = "";

    const answerSection = document.getElementById("answer-section");
    const answerText = document.getElementById("answer-text");
    const sourcesContainer = document.getElementById("sources-container");

    answerText.textContent = data.answer;

    // clear old sources
    sourcesContainer.innerHTML = "<h3>Sources</h3>";

    data.sources.forEach((src) => {
        const div = document.createElement("div");
        div.className = "source";
        div.textContent = `chunk ${src.chunk_index}: ${src.text}`;
        sourcesContainer.appendChild(div);
    });

    answerSection.style.display = "block";
}


//----------- CODE ------------//

// store the current session ID after a pdf is ingested
let currentSessionId = null;

// finds html elements by their id and stores them
const uploadForm = document.getElementById("upload-form"); // document represents the entire html page loaded by the browser
const uploadStatus = document.getElementById("upload-status");

// listens for when a form gets submitted
uploadForm.addEventListener("submit", async (event) => {
    event.preventDefault(); // stop browser reload

    uploadStatus.textContent = "Uploading and processing PDF...";
    uploadStatus.style.color = "#555";

    const fileInput = document.getElementById("pdf-file");
    if (!fileInput.files.length) {
        uploadStatus.textContent = "Please select a PDF file.";
        uploadStatus.style.color = "red";
        return;
    }

    // creates form data object and appends the PDF under the key "pdf_file" <- matches name in main.py
    const formData = new FormData()
    formData.append("pdf_file", fileInput.files[0]);

    // makes the api call
    try {
        const response = await fetch("/ingest", {
            method: "POST",
            body: formData,
        });

        // raises error from API call
        if(!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || "Upload failed");
        }

        // if successful, stores the data & handles the success
        const data = await response.json();
        handleIngestSuccess(data);

    }   catch (err) {
        uploadStatus.textContent = err.message;
        uploadStatus.style.color = "red";
    }
})



// handling the ask path
const askForm = document.getElementById("ask-form");
const askStatus = document.getElementById("ask-status");

askForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    if (!currentSessionId) {
        askStatus.textContent = "Please upload a PDF first.";
        askStatus.style.color = "red";
        return;
    }

    const questionInput = document.getElementById("question-input");
    const question = questionInput.value.trim();

    if (!question) {
        askStatus.textContent = "Please enter a question."
        askStatus.style.color = "red";
        return;
    }

    askStatus.textContent = "Thinking...";
    askStatus.style.color = "#555";

    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                session_id: currentSessionId,
                question: question,
            }),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || "Failed to get answer");
        }

        const data = await response.json();
        handleAskSuccess(data);

    } catch (err) {
        askStatus.textContent = err.message;
        askStatus.style.color = "red";
    }
});



