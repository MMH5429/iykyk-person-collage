# TensorFlow Lite keeps native entry points that the shrinker cannot see.
-keep class org.tensorflow.lite.** { *; }
-dontwarn org.tensorflow.lite.**
