using UnityEngine;
using UnityEngine.UI;

public class WebcamDisplay : MonoBehaviour
{
    [SerializeField] private RawImage webcamView;

    private WebCamTexture webcamTexture;

    private void Start()
    {
        if (webcamView == null)
        {
            Debug.LogError("No se asignó WebcamView.");
            return;
        }

        if (WebCamTexture.devices.Length == 0)
        {
            Debug.LogError("No se encontró ninguna cámara.");
            return;
        }

        WebCamDevice device = WebCamTexture.devices[0];

        webcamTexture = new WebCamTexture(device.name);

        webcamView.texture = webcamTexture;

        webcamTexture.Play();

        Debug.Log("Webcam iniciada: " + device.name);
    }

    private void OnDestroy()
    {
        if (webcamTexture != null && webcamTexture.isPlaying)
        {
            webcamTexture.Stop();
        }
    }
}
