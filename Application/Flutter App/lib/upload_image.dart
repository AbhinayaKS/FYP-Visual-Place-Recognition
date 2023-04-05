import 'dart:io';
import 'dart:convert';
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
  late File _image;
  late String _imageUrl;

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
    var request = http.MultipartRequest(
        'POST', Uri.parse('http://localhost:5000/nearestNeighbours'));

    var pic = await http.MultipartFile.fromPath('image', _image.path);
    request.files.add(pic);

    var response = await request.send();
    var bytes = await response.stream.toBytes();
    // var img = Image.memory(bytes);

    setState(() {
      _imageUrl = base64Encode(bytes);
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Image.file(_image),
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
            Image.memory(base64Decode(_imageUrl)),
          ],
        ),
      ),
    );
  }
}
