function ConfidenceBadge({ confidence = 0 }) {
    const percentage = Math.round(confidence * 100);
  
    let level = "low";
  
    if (confidence >= 0.8) {
      level = "high";
    } else if (confidence >= 0.6) {
      level = "medium";
    }
  
    return (
      <span className={`confidence-badge ${level}`}>
        {percentage}%
      </span>
    );
  }
  
  export default ConfidenceBadge;