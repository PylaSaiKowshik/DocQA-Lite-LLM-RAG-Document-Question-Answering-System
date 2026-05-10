<?php
set_time_limit(0);
ini_set('max_execution_time', 0);
error_reporting(0);

// ---- SSE STREAM MODE ----
if (isset($_GET['stream']) && isset($_GET['request_id'])) {
    $request_id = preg_replace('/[^a-z0-9]/', '', $_GET['request_id']);
    $cache_file = sys_get_temp_dir() . DIRECTORY_SEPARATOR . 'docmind_' . $request_id . '.jsonl';

    header('Content-Type: text/event-stream');
    header('Cache-Control: no-cache');
    header('X-Accel-Buffering: no');

    $debug_log = sys_get_temp_dir() . DIRECTORY_SEPARATOR . 'docmind_debug_' . $request_id . '.txt';
    file_put_contents($debug_log, "[" . date('H:i:s') . "] SSE started, waiting for: $cache_file\n");

    $last_pos     = 0;
    $timeout      = time() + 3600;
    $done         = false;
    $pending_done = false;
    $token_count  = 0;

    while (!$done && time() < $timeout) {
        if (file_exists($cache_file)) {
            clearstatcache(true, $cache_file);
            $content = @file_get_contents($cache_file);
            if ($content !== false && strlen($content) > $last_pos) {
                $new_content = substr($content, $last_pos);
                $last_pos    = strlen($content);
                $lines       = explode("\n", trim($new_content));

                file_put_contents($debug_log,
                    "[" . date('H:i:s') . "] Read " . count($lines) . " lines: " . substr($new_content, 0, 300) . "\n",
                    FILE_APPEND);

                foreach ($lines as $line) {
                    $line = trim($line);
                    if (!$line) continue;
                    $decoded = json_decode($line, true);
                    if (!$decoded) continue;

                    file_put_contents($debug_log,
                        "[" . date('H:i:s') . "] type=" . $decoded['type'] . " msg=" . substr($decoded['message'] ?? '', 0, 60) . "\n",
                        FILE_APPEND);

                    if ($decoded['type'] === 'done') {
                        $pending_done = true;
                        @unlink($cache_file);
                        continue;
                    }

                    if ($decoded['type'] === 'token') {
                        $token_count++;
                    }

                    echo "data: $line\n\n";
                    @ob_flush(); flush();
                }

                if ($pending_done) {
                    file_put_contents($debug_log,
                        "[" . date('H:i:s') . "] Sending done. Total tokens sent: $token_count\n",
                        FILE_APPEND);
                    usleep(150000);
                    echo "data: " . json_encode(['type' => 'done']) . "\n\n";
                    @ob_flush(); flush();
                    $done = true;
                    break;
                }
            }
        } else {
            static $last_wait_log = 0;
            if (time() - $last_wait_log >= 3) {
                file_put_contents($debug_log,
                    "[" . date('H:i:s') . "] Still waiting for cache file...\n",
                    FILE_APPEND);
                $last_wait_log = time();
            }
        }

        if (!$pending_done) {
            echo "data: " . json_encode(['type' => 'heartbeat']) . "\n\n";
            @ob_flush(); flush();
        }
        usleep(300000);
    }

    file_put_contents($debug_log,
        "[" . date('H:i:s') . "] SSE loop ended. done=$done tokens=$token_count\n",
        FILE_APPEND);
    exit;
}

// ---- NORMAL POST MODE ----
ob_clean();
header("Content-Type: application/json");

$data       = json_decode(file_get_contents("php://input"), true);
$question   = $data["question"]   ?? "";
$request_id = $data["request_id"] ?? "";
$filename   = $data["filename"]   ?? "";
$fileHash   = $data["file_hash"]  ?? "";

$uploadDir = "C:/xampp/htdocs/genai-summary+qa - kowshik/docmind_rag/saved_summaries/";

if ($fileHash) {
    $pdfFile = $fileHash . ".pdf";
} else if ($filename) {
    $pdfFile = basename($filename);
} else {
    $files = array_diff(scandir($uploadDir, SCANDIR_SORT_DESCENDING), ['.', '..']);
    $pdfFile = '';
    foreach ($files as $file) {
        if (strtolower(pathinfo($file, PATHINFO_EXTENSION)) === 'pdf') {
            $pdfFile = $file;
            break;
        }
    }
}

if ($pdfFile && $question) {
    $pdfPath = $uploadDir . $pdfFile;

    if (!file_exists($pdfPath)) {
        echo json_encode(["answer" => "❌ PDF file not found: " . $pdfFile]);
        exit;
    }

    $postData = json_encode([
        "pdf_path"   => $pdfPath,
        "question"   => $question,
        "request_id" => $request_id,
        "file_hash"  => $fileHash
    ]);

    $ch = curl_init("http://127.0.0.1:5000/process");
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_POST,           true);
    curl_setopt($ch, CURLOPT_HTTPHEADER,     ["Content-Type: application/json"]);
    curl_setopt($ch, CURLOPT_POSTFIELDS,     $postData);
    curl_setopt($ch, CURLOPT_TIMEOUT,        0);
    curl_setopt($ch, CURLOPT_CONNECTTIMEOUT, 30);

    $response = curl_exec($ch);

    $debug_log = sys_get_temp_dir() . DIRECTORY_SEPARATOR . 'docmind_debug_' . $request_id . '.txt';
    $decoded_response = json_decode($response, true);
    file_put_contents($debug_log,
        "[" . date('H:i:s') . "] POST answer='" . substr($decoded_response['answer'] ?? 'NULL', 0, 100) . "'\n",
        FILE_APPEND);

    if ($response === false) {
        echo json_encode(["answer" => "❌ Flask server not running. Start app.py first."]);
        curl_close($ch);
        exit;
    }

    curl_close($ch);
    echo $response;
    exit;

} else {
    echo json_encode(["answer" => "❌ No PDF found or question missing."]);
    exit;
}
?>