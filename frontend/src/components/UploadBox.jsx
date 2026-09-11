import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { extractPII } from "../api/piiApi";

const ACCEPTED_TYPES = [
  ".pdf",
  ".docx",
  ".xlsx",
  ".xls",
  ".png",
  ".jpg",
  ".jpeg",
];

function UploadBox() {
  const inputRef = useRef(null);
  const [files, setFiles] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const navigate = useNavigate();
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState("");

  const addFiles = (selectedFiles) => {
    const validFiles = Array.from(selectedFiles).filter((file) => {
      const extension = "." + file.name.split(".").pop().toLowerCase();
      return ACCEPTED_TYPES.includes(extension);
    });

    setFiles((current) => {
      const existingNames = new Set(current.map((file) => file.name));

      const newFiles = validFiles.filter(
        (file) => !existingNames.has(file.name),
      );

      return [...current, ...newFiles];
    });
  };

  const handleInputChange = (event) => {
    addFiles(event.target.files);
    event.target.value = "";
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);

    addFiles(event.dataTransfer.files);
  };

  const removeFile = (fileName) => {
    setFiles((current) => current.filter((file) => file.name !== fileName));
  };

  const formatSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;

    if (bytes < 1024 * 1024) {
      return `${(bytes / 1024).toFixed(1)} KB`;
    }

    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const handleAnalyze = () => {
    if (files.length === 0 || isAnalyzing) return;

    setError("");
    setIsAnalyzing(true);

    sessionStorage.removeItem("piiExtractionResults");
    sessionStorage.removeItem("piiExtractionError");

    sessionStorage.setItem(
      "piiExtractionStatus",
      JSON.stringify({
        status: "processing",
        documents: files.map((file) => file.name),
        startedAt: Date.now(),
      }),
    );

    navigate("/results");

    extractPII(files)
      .then((result) => {
        sessionStorage.setItem(
          "piiExtractionResults",
          JSON.stringify(result),
        );

        sessionStorage.setItem(
          "piiExtractionStatus",
          JSON.stringify({
            status: "complete",
            documents: files.map((file) => file.name),
          }),
        );
      })
      .catch((err) => {
        console.error(err);

        sessionStorage.setItem(
          "piiExtractionError",
          err.message || "Failed to analyze documents.",
        );

        sessionStorage.setItem(
          "piiExtractionStatus",
          JSON.stringify({
            status: "error",
          }),
        );
      })
      .finally(() => {
        setIsAnalyzing(false);
      });
  };

  return (
    <div className="upload-container">
      <div
        className={`drop-zone ${isDragging ? "dragging" : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={ACCEPTED_TYPES.join(",")}
          onChange={handleInputChange}
          hidden
        />

        <div className="upload-icon">↑</div>

        <h2>Drop your documents here</h2>

        <p>
          or <span>browse files</span> from your computer
        </p>

        <div className="supported-files">
          PDF · DOCX · XLSX · XLS · PNG · JPG · JPEG
        </div>
      </div>

      {files.length > 0 && (
        <div className="selected-files">
          <div className="files-header">
            <div>
              <h3>Selected documents</h3>
              <span>
                {files.length} document{files.length !== 1 ? "s" : ""}
              </span>
            </div>

            <button className="clear-button" onClick={() => setFiles([])}>
              Clear all
            </button>
          </div>

          <div className="files-list">
            {files.map((file) => (
              <div className="file-item" key={file.name}>
                <div className="file-type">
                  {file.name.split(".").pop().toUpperCase()}
                </div>

                <div className="file-info">
                  <strong>{file.name}</strong>
                  <span>{formatSize(file.size)}</span>
                </div>

                <button
                  className="remove-file"
                  onClick={(event) => {
                    event.stopPropagation();
                    removeFile(file.name);
                  }}
                >
                  ×
                </button>
              </div>
            ))}
          </div>

          <button
            className="analyze-button"
            disabled={isAnalyzing}
            onClick={(event) => {
              event.stopPropagation();
              handleAnalyze();
            }}
          >
            {isAnalyzing ? "Analyzing..." : "Analyze Documents"}
            {!isAnalyzing && <span>→</span>}
          </button>

          {error && <div className="upload-error">{error}</div>}
        </div>
      )}
    </div>
  );
}

export default UploadBox;