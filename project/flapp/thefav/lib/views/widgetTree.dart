import 'package:flutter/material.dart';
import 'package:thefav/data/identifier.dart';
import 'package:thefav/views/pages/popuppage.dart';
// import 'package:thefav/views/pages/popuppage.dart';

import 'widget/HomeScreen.dart';

class widgetTree extends StatefulWidget {
  const widgetTree({super.key});

  @override
  State<widgetTree> createState() => _widgetTreeState();
}

class _widgetTreeState extends State<widgetTree> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Center(
            child: Text('The Fav', style: TextStyle(color: Colors.black))),
        backgroundColor: Colors.purple.shade200,
        actions: [
          Center(
            child: ValueListenableBuilder(
                valueListenable: isModeNotifier,
                builder: (context, isMode, child) {
                  return Padding(
                    padding: EdgeInsets.only(right: 20.0),
                    child: IconButton(
                      color: Colors.black,
                      onPressed: () {
                        isModeNotifier.value = !isModeNotifier.value;
                      },
                      icon: isMode
                          ? Icon(Icons.dark_mode)
                          : Icon(Icons.light_mode),
                    ),
                  );
                }),
          ),
        ],
      ),
      body: HomeScreen(),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          showDialog(
            context: context,
            builder: (context) {
              return AlertDialog(
                title: Text("Select File Type"),
                content: Column(
                  mainAxisSize: MainAxisSize.max,
                  
                  children: [
                    ElevatedButton(
                      onPressed: () {
                        Navigator.pop(context); // Close popup
                      },
                      child: Text("Add Image"),
                    ),
                    ElevatedButton(
                      onPressed: () {
                        Navigator.pop(context); // Close popup
                      },
                      child: Text("Add Document"),
                    ),
                  ],
                ),
              );
            },
          );
        },
        child: Icon(Icons.add),
      ),
    );
  }
}
