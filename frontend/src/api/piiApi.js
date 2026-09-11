const API_BASE_URL = "http://127.0.0.1:8000";

export async function extractPII(files) {
  const formData = new FormData();

  for (const file of files) {
    formData.append("files", file);
  }

  const response = await fetch(
    `${API_BASE_URL}/extract`,
    {
      method: "POST",
      body: formData,
    }
  );

  if (!response.ok) {
    const errorText = await response.text();

    throw new Error(
      errorText || "PII extraction failed"
    );
  }

  return await response.json();
}