import 'dart:io';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

class UploadPhoto extends StatefulWidget {
  const UploadPhoto({Key? key, required this.title}) : super(key: key);

  final String title;

  @override
  MyHomePageState createState() => MyHomePageState();
}

class MyHomePageState extends State<UploadPhoto> {
  File? _image;
  Uint8List? _imageUrl;

  final _picker = ImagePicker();

  Future getImage() async {
    final pickedFile = await _picker.pickImage(source: ImageSource.gallery);

    setState(() {
      if (pickedFile != null) {
        _image = File(pickedFile.path);
      } else {
        debugPrint('No image selected.');
      }
    });
  }

  Future getNearestNeighbours() async {
    if (_image == null) {
      debugPrint('No image selected.');
      return;
    }

    var request = http.MultipartRequest(
        'POST', Uri.parse('http://192.168.0.103:5000/nearestNeighbours'));

    var pic = await http.MultipartFile.fromPath('image', _image!.path);
    request.files.add(pic);
    request.headers.addAll({'Accept': 'image/png', 'Connection': 'Keep-Alive'});
    var response = await request.send();
    debugPrint('Response: ${response.headers}');
    // var response = await http
    //     .get(Uri.parse('http://192.168.0.103:5000/nearestNeighbours'));
    if (response.statusCode == 200) {
      var bytes = await response.stream.toBytes();
      setState(() {
        _imageUrl = bytes;
      });
    } else {
      debugPrint('Error: ${response.statusCode}');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
      ),
      body: SingleChildScrollView(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            _image != null ? Image.file(_image!) : Container(),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: getImage,
              child: const Text('Select an image'),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: getNearestNeighbours,
              child: const Text('Get nearest neighbours'),
            ),
            const SizedBox(height: 20),
            _imageUrl != null ? Image.memory(_imageUrl!) : Container(),
          ],
        ),
      ),
    );
  }
}
