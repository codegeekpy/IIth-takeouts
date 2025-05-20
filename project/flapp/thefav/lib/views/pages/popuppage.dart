import 'package:flutter/material.dart';
import 'package:thefav/views/widget/HomeScreen.dart';

class PopUpPage extends StatefulWidget {
  const PopUpPage({super.key});

  @override
  State<PopUpPage> createState() => _PopUpPageState();
}

class _PopUpPageState extends State<PopUpPage> {
  @override
  Widget build(BuildContext context) {
    return HomeScreen();
  }
}
