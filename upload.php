<?php
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    if (isset($_FILES["file"])) {
        $targetDir = "docmind_rag/saved_summaries/";
        $fileName = basename($_FILES["file"]["name"]);
        $fileType = strtolower(pathinfo($fileName, PATHINFO_EXTENSION));

        if ($fileType !== "pdf") {
            echo json_encode(["status" => "error", "message" => "Only PDF files are allowed."]);
            exit;
        }

        if (!is_dir($targetDir)) {
            mkdir($targetDir, 0777, true);
        }

        if ($_FILES["file"]["error"] !== 0) {
            echo json_encode(["status" => "error", "message" => "Upload error: " . $_FILES["file"]["error"]]);
            exit;
        }

        // Clean up session-only PDFs (those with no matching .txt = user chose "No" previously)
        foreach (glob($targetDir . "*.pdf") as $oldPdf) {
            $oldTxt = str_replace(".pdf", ".txt", $oldPdf);
            if (!file_exists($oldTxt)) {
                unlink($oldPdf);
            }
        }

        // Save temporarily to compute hash
        $tmpPath = $targetDir . $fileName;
        if (move_uploaded_file($_FILES["file"]["tmp_name"], $tmpPath)) {
            // Rename to hash-based filename
            $hash = md5_file($tmpPath);
            $hashedPath = $targetDir . $hash . ".pdf";
            rename($tmpPath, $hashedPath);
            $fullPath = realpath($hashedPath);

            echo json_encode([
                "status"   => "success",
                "filename" => $fileName,
                "path"     => $fullPath,
                "hash"     => $hash
            ]);
        } else {
            echo json_encode(["status" => "error", "message" => "Failed to move uploaded file."]);
        }
    } else {
        echo json_encode(["status" => "error", "message" => "No file uploaded."]);
    }
} else {
    echo json_encode(["status" => "error", "message" => "Invalid request."]);
}
?>