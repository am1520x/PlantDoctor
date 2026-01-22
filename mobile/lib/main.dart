import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

import 'package:mime/mime.dart';
import 'package:http_parser/http_parser.dart';
import 'package:path/path.dart' as p;


void main() => runApp(const PlantDoctorApp());

class PlantDoctorApp extends StatelessWidget {
  const PlantDoctorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'PlantDoctor',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.green),
        useMaterial3: true,
      ),
      home: const PlantIdPage(),
    );
  }
}

/// -------- CONFIG --------
/// You can override these without changing code:
/// flutter run --dart-define=API_BASE_URL=... --dart-define=API_PREDICT_PATH=...
class ApiConfig {
  static const String baseUrl = String.fromEnvironment(
    'API_BASE_URL',
    defaultValue: 'https://plantdoctor-zz79.onrender.com',
  );

  /// IMPORTANT: set this to match your FastAPI endpoint path.
  /// Common: /predict
  static const String predictPath = String.fromEnvironment(
    'API_PREDICT_PATH',
    defaultValue: '/predict',
  );

  /// IMPORTANT: set to whatever your FastAPI expects:
  /// e.g. file: UploadFile = File(...) -> "file"
  /// e.g. image: UploadFile = File(...) -> "image"
  static const String fileFieldName = String.fromEnvironment(
    'API_FILE_FIELD',
    defaultValue: 'file',
  );

  /// Add headers here if you add auth later (API keys, etc.).
  static const Map<String, String> extraHeaders = {};
}

class Prediction {
  final String label;
  final double confidence;

  const Prediction({required this.label, required this.confidence});

  factory Prediction.fromJson(Map<String, dynamic> json) {
    // Supports a few common response shapes:
    // { "class": "...", "confidence": 0.93 }
    // { "label": "...", "score": 0.93 }
    // { "prediction": "...", "probability": 0.93 }
    final labelAny = json['class'] ?? json['label'] ?? json['prediction'] ?? json['plant_class'];
    if (labelAny == null) {
      throw const FormatException('Response JSON missing label field (class/label/prediction).');
    }

    final confAny = json['confidence'] ??
        json['score'] ??
        json['probability'] ??
        json['confidence_score'];

    if (confAny == null) {
      throw const FormatException('Response JSON missing confidence field (confidence/score/etc).');
    }

    final conf = (confAny is num)
        ? confAny.toDouble()
        : double.tryParse(confAny.toString());

    if (conf == null) {
      throw const FormatException('Could not parse confidence as a number.');
    }

    return Prediction(label: labelAny.toString(), confidence: conf);
  }
}

Future<http.MultipartFile> buildImagePart(File imageFile) async {
  final mimeType = lookupMimeType(imageFile.path) ?? 'image/jpeg';
  final mediaType = MediaType.parse(mimeType);

  return http.MultipartFile.fromPath(
    ApiConfig.fileFieldName,   // "file"
    imageFile.path,
    contentType: mediaType,    // <-- fixes the 415
    filename: p.basename(imageFile.path),
  );
}


class PlantApiClient {
  Future<Prediction> predict(File imageFile) async {
    final uri = Uri.parse('${ApiConfig.baseUrl}${ApiConfig.predictPath}');
    final request = http.MultipartRequest('POST', uri);

    request.headers.addAll(ApiConfig.extraHeaders);

    request.files.add(await buildImagePart(imageFile));

    final streamed = await request.send();
    final body = await streamed.stream.bytesToString();

    if (streamed.statusCode < 200 || streamed.statusCode >= 300) {
      throw HttpException('HTTP ${streamed.statusCode}: $body');
    }

    final decoded = jsonDecode(body);
    if (decoded is! Map<String, dynamic>) {
      throw const FormatException('Response JSON was not an object.');
    }
    return Prediction.fromJson(decoded);
  }
}

class PlantIdPage extends StatefulWidget {
  const PlantIdPage({super.key});

  @override
  State<PlantIdPage> createState() => _PlantIdPageState();
}

class _PlantIdPageState extends State<PlantIdPage> {
  final _picker = ImagePicker();
  final _api = PlantApiClient();

  File? _image;
  Prediction? _prediction;
  String? _error;
  bool _loading = false;

  Future<void> _pick(ImageSource source) async {
    setState(() {
      _error = null;
      _prediction = null;
    });

    final picked = await _picker.pickImage(
      source: source,
      imageQuality: 92,
      maxWidth: 1600,
    );
    if (picked == null) return;

    setState(() => _image = File(picked.path));
    await _submit();
  }

  Future<void> _submit() async {
    final img = _image;
    if (img == null) {
      setState(() => _error = 'Pick an image first.');
      return;
    }

    setState(() {
      _loading = true;
      _error = null;
      _prediction = null;
    });

    try {
      final pred = await _api.predict(img);
      if (!mounted) return;
      setState(() => _prediction = pred);
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (!mounted) return;
      setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final pred = _prediction;

    return Scaffold(
      appBar: AppBar(
        title: const Text('PlantDoctor'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            Expanded(
              child: Center(
                child: _image == null
                    ? const Text('Pick a plant photo to identify it.')
                    : ClipRRect(
                        borderRadius: BorderRadius.circular(16),
                        child: Image.file(_image!, fit: BoxFit.contain),
                      ),
              ),
            ),
            const SizedBox(height: 12),

            if (_loading) const LinearProgressIndicator(),
            if (_error != null) ...[
              const SizedBox(height: 8),
              Text(
                _error!,
                style: TextStyle(color: Theme.of(context).colorScheme.error),
                maxLines: 4,
                overflow: TextOverflow.ellipsis,
              ),
              const SizedBox(height: 6),
              Text(
                'Config: ${ApiConfig.baseUrl}${ApiConfig.predictPath} (field="${ApiConfig.fileFieldName}")',
                style: Theme.of(context).textTheme.bodySmall,
              ),
            ],
            if (pred != null) ...[
              const SizedBox(height: 8),
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Row(
                    children: [
                      const Icon(Icons.local_florist),
                      const SizedBox(width: 10),
                      Expanded(
                        child: Text(
                          pred.label,
                          style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
                        ),
                      ),
                      Text('${(pred.confidence * 100).toStringAsFixed(1)}%'),
                    ],
                  ),
                ),
              ),
            ],

            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: FilledButton.icon(
                    onPressed: _loading ? null : () => _pick(ImageSource.camera),
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('Camera'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: FilledButton.icon(
                    onPressed: _loading ? null : () => _pick(ImageSource.gallery),
                    icon: const Icon(Icons.photo_library),
                    label: const Text('Gallery'),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 10),
            SizedBox(
              width: double.infinity,
              child: OutlinedButton.icon(
                onPressed: (_loading || _image == null) ? null : _submit,
                icon: const Icon(Icons.cloud_upload),
                label: const Text('Re-try upload'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
