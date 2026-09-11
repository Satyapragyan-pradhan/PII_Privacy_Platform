import { useEffect, useState } from "react";
import ConfidenceBadge from "../components/ConfidenceBadge";

const ANALYSIS_STAGES = [
  {
    id: "ingestion",
    label: "Document ingestion",
    description: "Reading uploaded documents",
  },
  {
    id: "ocr",
    label: "OCR — PaddleOCR",
    description: "Extracting text from document images",
  },
  {
    id: "regex",
    label: "Structured extraction — Regex",
    description: "Detecting structured PII patterns",
  },
  {
    id: "deberta",
    label: "Semantic extraction — DeBERTa",
    description: "Understanding contextual entities",
  },
  {
    id: "reconciliation",
    label: "Candidate reconciliation",
    description: "Merging and validating candidates",
  },
  {
    id: "context",
    label: "Contextual validation",
    description: "Checking ambiguous information",
  },
  {
    id: "confidence",
    label: "Confidence scoring",
    description: "Calculating final confidence",
  },
];

function AnalysisAnimation() {
  const [activeStage, setActiveStage] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveStage((current) => {
        if (current >= ANALYSIS_STAGES.length - 1) {
          return current;
        }

        return current + 1;
      });
    }, 1200);

    return () => clearInterval(interval);
  }, []);

  const processingStatus = sessionStorage.getItem(
    "piiExtractionStatus",
  );

  let documentCount = 1;

  if (processingStatus) {
    try {
      const status = JSON.parse(processingStatus);

      documentCount = status.documents?.length || 1;
    } catch {
      documentCount = 1;
    }
  }

  return (
    <div className="analysis-container">
      <div className="analysis-header">
        <div className="analysis-spinner"></div>

        <div>
          <p className="section-label">DOCUMENT ANALYSIS</p>

          <h1>Analyzing your documents</h1>

          <p className="page-description">
            The PII Privacy Platform is processing{" "}
            {documentCount} document
            {documentCount !== 1 ? "s" : ""}.
          </p>
        </div>
      </div>

      <div className="analysis-pipeline">
        {ANALYSIS_STAGES.map((stage, index) => {
          const completed = index < activeStage;
          const active = index === activeStage;

          return (
            <div
              className={`analysis-stage ${
                completed ? "completed" : ""
              } ${active ? "active" : ""}`}
              key={stage.id}
            >
              <div className="stage-indicator">
                {completed ? (
                  "✓"
                ) : active ? (
                  <span className="stage-loader"></span>
                ) : (
                  index + 1
                )}
              </div>

              <div className="stage-content">
                <strong>{stage.label}</strong>

                <span>{stage.description}</span>
              </div>
            </div>
          );
        })}
      </div>

      <div className="analysis-note">
        <span className="analysis-pulse"></span>
        AI-assisted extraction in progress
      </div>
    </div>
  );
}

function ResultsPage() {
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("processing");
  const [error, setError] = useState("");

  useEffect(() => {
    const checkStatus = () => {
      const storedResult = sessionStorage.getItem(
        "piiExtractionResults",
      );

      const storedStatus = sessionStorage.getItem(
        "piiExtractionStatus",
      );

      const storedError = sessionStorage.getItem(
        "piiExtractionError",
      );

      if (storedResult) {
        try {
          setResult(JSON.parse(storedResult));
          setStatus("complete");
          return;
        } catch {
          console.error("Invalid extraction result.");
        }
      }

      if (storedStatus) {
        try {
          const parsedStatus = JSON.parse(storedStatus);

          setStatus(parsedStatus.status || "processing");
        } catch {
          setStatus("processing");
        }
      }

      if (storedError) {
        setError(storedError);
      }
    };

    checkStatus();

    const interval = setInterval(checkStatus, 300);

    return () => clearInterval(interval);
  }, []);

  if (status === "processing" && !result) {
    return (
      <div className="results-page">
        <AnalysisAnimation />
      </div>
    );
  }

  if (status === "error") {
    return (
      <div className="results-page">
        <p className="section-label">EXTRACTION RESULTS</p>

        <h1>Analysis failed</h1>

        <p className="page-description">
          {error || "Something went wrong while processing the documents."}
        </p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="results-page">
        <p className="section-label">EXTRACTION RESULTS</p>

        <h1>No results available</h1>

        <p className="page-description">
          Upload and analyze a document to view detected PII.
        </p>
      </div>
    );
  }

  const documents = result.documents || [];

  return (
    <div className="results-page">
      <div className="results-heading">
        <div>
          <p className="section-label">EXTRACTION RESULTS</p>

          <h1>Detected PII</h1>

          <p className="page-description">
            Review the sensitive information identified across your
            documents.
          </p>
        </div>

        <div className="result-status">
          <span className="status-dot"></span>
          Analysis complete
        </div>
      </div>

      <div className="result-summary">
        <div className="summary-card">
          <span>Documents</span>
          <strong>
            {result.documents_processed ?? documents.length}
          </strong>
        </div>

        <div className="summary-card">
          <span>Total entities</span>
          <strong>
            {result.analytics?.total_entities ?? 0}
          </strong>
        </div>

        <div className="summary-card">
          <span>High confidence</span>
          <strong>
            {result.analytics?.high_confidence ?? 0}
          </strong>
        </div>

        <div className="summary-card">
          <span>Medium confidence</span>
          <strong>
            {result.analytics?.medium_confidence ?? 0}
          </strong>
        </div>

        <div className="summary-card">
          <span>Processing time</span>
          <strong>
            {result.processing_time_ms
              ? `${(result.processing_time_ms / 1000).toFixed(2)}s`
              : "—"}
          </strong>
        </div>
      </div>

      {documents.map((document, documentIndex) => {
        const entities = document.entities || [];
        const analytics = document.analytics || {};

        return (
          <div
            className="results-card"
            key={`${document.document}-${documentIndex}`}
          >
            <div className="results-card-header">
              <div>
                <h2>{document.document || "Document"}</h2>

                <span>
                  {analytics.total_entities ?? entities.length}{" "}
                  detected entities
                  {" • "}
                  {document.processing_time_ms
                    ? `${(
                        document.processing_time_ms / 1000
                      ).toFixed(2)}s`
                    : "—"}
                </span>
              </div>

              <div className="document-confidence-summary">
                <span>
                  High: {analytics.high_confidence ?? 0}
                </span>

                <span>
                  Medium: {analytics.medium_confidence ?? 0}
                </span>

                <span>
                  Low: {analytics.low_confidence ?? 0}
                </span>
              </div>
            </div>

            {entities.length === 0 ? (
              <div className="empty-results">
                No PII entities were detected.
              </div>
            ) : (
              <div className="results-table-wrapper">
                <table className="results-table">
                  <thead>
                    <tr>
                      <th>PII TYPE</th>
                      <th>VALUE</th>
                      <th>SOURCE</th>
                      <th>CONFIDENCE</th>
                      <th>VALIDATION</th>
                    </tr>
                  </thead>

                  <tbody>
                    {entities.map((entity, index) => (
                      <tr
                        key={`${entity.type}-${entity.value}-${index}`}
                      >
                        <td>
                          <span className="entity-type">
                            {entity.type}
                          </span>
                        </td>

                        <td>
                          <span className="entity-value">
                            {entity.value}
                          </span>
                        </td>

                        <td>
                          <span className="source-badge">
                            {entity.source || "—"}
                          </span>
                        </td>

                        <td>
                          <ConfidenceBadge
                            confidence={entity.confidence}
                          />
                        </td>

                        <td>
                          {entity.validated ? (
                            <span className="validation valid">
                              ✓ Valid
                            </span>
                          ) : (
                            <span className="validation">
                              Not validated
                            </span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

export default ResultsPage;