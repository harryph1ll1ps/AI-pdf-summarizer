// store the current session ID after a pdf is ingested
let currentSessionId = null;

// finds html elements by their id and stores them
const uploadForm = document.getElementById("upload-form");
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


function handleIngestSuccess(data) {
    currentSessionId = data.session_id; // essentially connects the current webpage to that document
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
}