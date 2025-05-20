import 'package:flutter/material.dart';

import '../pages/linkspage.dart';
import '../pages/moviepage.dart';
import '../pages/musicpage.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(20.0),
      child: ListView(
        children: [
          Wrap(
            spacing: 20.0,
            runSpacing: 20.0,
            alignment: WrapAlignment.center,
            children: [
              Padding(
                padding: const EdgeInsets.all(20.0),
                child: GestureDetector(
                  onTap: () {
                    Navigator.push(context, MaterialPageRoute(
                      builder: (context) {
                        return MusicPage();
                      },
                    ));
                  },
                  child: Container(
                    height: 200,
                    width: 200,
                    decoration: BoxDecoration(
                      color: Colors.purple.shade100,
                      borderRadius: BorderRadius.circular(20.0),
                    ),
                    child: Center(
                        child: Text(
                      "Music",
                      style: TextStyle(color: Colors.black),
                    )),
                  ),
                ),
              ),
              Padding(
                padding: const EdgeInsets.all(20.0),
                child: GestureDetector(
                  onTap: () {
                    Navigator.push(context, MaterialPageRoute(
                      builder: (context) {
                        return MoviePage();
                      },
                    ));
                  },
                  child: Container(
                    height: 200,
                    width: 200,
                    decoration: BoxDecoration(
                        color: Colors.amber.shade100,
                        borderRadius: BorderRadius.circular(20.0)),
                    child: Center(
                        child: Text(
                      "Movie",
                      style: TextStyle(color: Colors.black),
                    )),
                  ),
                ),
              ),
              Padding(
                padding: const EdgeInsets.all(20.0),
                child: GestureDetector(
                  onTap: () {
                    Navigator.push(context, MaterialPageRoute(
                      builder: (context) {
                        return LinksPage();
                      },
                    ));
                  },
                  child: Container(
                    height: 200,
                    width: 200,
                    decoration: BoxDecoration(
                      color: Colors.green.shade100,
                      borderRadius: BorderRadius.circular(20.0),
                    ),
                    child: Center(
                        child: Text(
                      "Link",
                      style: TextStyle(color: Colors.black),
                    )),
                  ),
                ),
              ),
              Padding(
                padding: const EdgeInsets.all(20.0),
                child: GestureDetector(
                  onTap: () {
                    Navigator.push(context, MaterialPageRoute(
                      builder: (context) {
                        return MusicPage();
                      },
                    ));
                  },
                  child: Container(
                    height: 200,
                    width: 200,
                    decoration: BoxDecoration(
                        color: Colors.blue.shade100,
                        borderRadius: BorderRadius.circular(20.0)),
                    child: Center(
                        child: Text("List page",
                            style: TextStyle(color: Colors.black))),
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
