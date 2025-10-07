using UnityEngine;

namespace WorldNavigator.Animation
{
    /// <summary>
    /// Simple wave animation for water surfaces
    /// </summary>
    public class WaveAnimation : MonoBehaviour
    {
        [Header("Wave Settings")]
        [SerializeField] private float waveHeight = 0.2f;
        [SerializeField] private float waveSpeed = 1f;
        [SerializeField] private float waveFrequency = 1f;
        
        private Vector3 originalPosition;
        private float timeOffset;
        
        private void Start()
        {
            originalPosition = transform.localPosition;
            timeOffset = Random.Range(0f, 2f * Mathf.PI);
        }
        
        private void Update()
        {
            float wave = Mathf.Sin((Time.time + timeOffset) * waveSpeed) * waveHeight;
            transform.localPosition = originalPosition + Vector3.up * wave;
            
            // Add gentle rotation for water movement effect
            transform.Rotate(Vector3.up, waveFrequency * Time.deltaTime, Space.Self);
        }
    }
}