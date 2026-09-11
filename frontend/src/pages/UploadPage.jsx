import UploadBox from "../components/UploadBox";

function UploadPage() {
  return (
    <div className="upload-page">
      <div className="page-heading">
        <div>
          <p className="section-label">DOCUMENT ANALYSIS</p>

          <h1>Analyze your documents</h1>

          <p className="page-description">
            Upload documents and let the PII Privacy Platform identify,
            validate, and classify sensitive personal information.
          </p>
        </div>
      </div>

      <UploadBox />
    </div>
  );
}

export default UploadPage;