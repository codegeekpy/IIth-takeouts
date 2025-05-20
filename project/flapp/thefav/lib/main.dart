import 'package:flutter/material.dart';

import 'data/identifier.dart';
import 'views/widgettree.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder(
      valueListenable: isModeNotifier,
      builder: (context, isMode, child) {
        return MaterialApp(
          debugShowCheckedModeBanner: false,
          theme: ThemeData(
            colorScheme: ColorScheme.fromSeed(
                seedColor: Colors.purple.shade400,
                brightness: isMode ? Brightness.dark : Brightness.light),
          ),
          home: widgetTree(),
        );
      },
    );
  }
}
