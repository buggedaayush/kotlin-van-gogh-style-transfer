import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import java.io.File
import java.nio.FloatBuffer
import javax.imageio.ImageIO
import kotlin.math.roundToInt

// Longest side (in pixels) the photo is shrunk to before styling.
// Bigger = more detail but slower and more memory. 768 is a good starting point.
const val MAX_SIDE = 768

fun main(args: Array<String>) {
    val inputPath = args.getOrElse(0) { "images/city.jpg" }
    val outputPath = args.getOrElse(1) { "output/city_vangogh.png" }
    val modelPath = "models/starry-night.onnx"

    // 1. Load the photo and resize it so width and height are multiples of 4
    val original = ImageIO.read(File(inputPath))
        ?: error("Could not read '$inputPath' (use a .jpg or .png file)")
    val photo = resizeForModel(original, MAX_SIDE)
    println("Photo resized to ${photo.width} x ${photo.height}")

    // 2. Load the style model (this is the "Van Gogh" brain)
    val env = OrtEnvironment.getEnvironment()
    env.createSession(modelPath, OrtSession.SessionOptions()).use { session ->

        // 3. Run the photo through the model
        val start = System.currentTimeMillis()
        val result = stylize(env, session, imageToFloats(photo), photo.width, photo.height)
        println("Stylized in ${System.currentTimeMillis() - start} ms")

        // 4. Turn the model output back into a picture and save it
        File(outputPath).absoluteFile.parentFile?.mkdirs()
        ImageIO.write(floatsToImage(result, photo.width, photo.height), "png", File(outputPath))
        println("Saved: ${File(outputPath).absolutePath}")
    }
}

/** Shrinks the image (never enlarges) and makes width/height multiples of 4, which this model requires. */
fun resizeForModel(src: BufferedImage, maxSide: Int): BufferedImage {
    val scale = minOf(1.0, maxSide.toDouble() / maxOf(src.width, src.height))
    val w = maxOf(4, (src.width * scale).toInt() / 4 * 4)
    val h = maxOf(4, (src.height * scale).toInt() / 4 * 4)
    val out = BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
    val g = out.createGraphics()
    g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC)
    g.drawImage(src, 0, 0, w, h, null)
    g.dispose()
    return out
}

/** Image -> flat float array laid out as [R plane][G plane][B plane], values 0..255 (what the model expects). */
fun imageToFloats(img: BufferedImage): FloatArray {
    val w = img.width
    val h = img.height
    val plane = w * h
    val pixels = img.getRGB(0, 0, w, h, null, 0, w)
    val data = FloatArray(3 * plane)
    for (i in 0 until plane) {
        val p = pixels[i]
        data[i] = ((p shr 16) and 0xFF).toFloat()
        data[plane + i] = ((p shr 8) and 0xFF).toFloat()
        data[2 * plane + i] = (p and 0xFF).toFloat()
    }
    return data
}

/** Runs the model. Input and output shape: [1, 3, height, width]. */
fun stylize(env: OrtEnvironment, session: OrtSession, data: FloatArray, w: Int, h: Int): FloatArray {
    val inputName = session.inputNames.first()
    val shape = longArrayOf(1, 3, h.toLong(), w.toLong())
    return OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape).use { input ->
        session.run(mapOf(inputName to input)).use { output ->
            val buffer = (output[0] as OnnxTensor).floatBuffer
            FloatArray(buffer.remaining()).also { buffer.get(it) }
        }
    }
}

/** Flat [R plane][G plane][B plane] floats -> image. Values outside 0..255 are clamped (the model overshoots a bit). */
fun floatsToImage(data: FloatArray, w: Int, h: Int): BufferedImage {
    val plane = w * h
    val pixels = IntArray(plane)
    for (i in 0 until plane) {
        val r = data[i].coerceIn(0f, 255f).roundToInt()
        val g = data[plane + i].coerceIn(0f, 255f).roundToInt()
        val b = data[2 * plane + i].coerceIn(0f, 255f).roundToInt()
        pixels[i] = (r shl 16) or (g shl 8) or b
    }
    val img = BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
    img.setRGB(0, 0, w, h, pixels, 0, w)
    return img
}
