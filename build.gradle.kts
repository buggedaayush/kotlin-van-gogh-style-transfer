plugins {
    kotlin("jvm") version "2.4.20"
}

group = "org.example"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(kotlin("test"))
    implementation("com.microsoft.onnxruntime:onnxruntime:1.27.0")
}

kotlin {
    jvmToolchain(26)
}

tasks.test {
    useJUnitPlatform()
}