/*
1. Install Java JDK 17 from: [Eclipse Temurin JDK 17](https://adoptium.net/temurin/releases/?version=17&utm_source=chatgpt.com)
2. Install the `.msi` file normally.
3. Set `JAVA_HOME` to your JDK path.
4. Add `%JAVA_HOME%\bin` to PATH.
5. Remove old Oracle Java paths from PATH.
6. Restart CMD.
7. Verify Java using:
```
java -version
javac -version
```
8. Install Scala from: [Scala Official Download](https://www.scala-lang.org/download/?utm_source=chatgpt.com)
9. During install, type `y` to add Scala to PATH.
10. Restart CMD again.
11. Verify Scala using:
```
scala -version
scalac -version
```
12. Create `12.scala`.
13. Compile program:
```
scalac 12.scala
```
14. Run program:
```
scala 12.scala
```
15. Output:

```
Hello World!
```
*/


object HelloWorld {

  def main(args: Array[String]): Unit = {

    println("Hello World!")

  }

}