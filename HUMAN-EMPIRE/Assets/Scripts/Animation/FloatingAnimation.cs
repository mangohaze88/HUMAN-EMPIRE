using UnityEngine;

namespace WorldNavigator.Animation
{
    /// <summary>
    /// Provides gentle floating animation for land objects
    /// </summary>
    public class FloatingAnimation : MonoBehaviour
    {
        [Header("Float Settings")]
        [SerializeField] private float floatAmplitude = 0.5f;
        [SerializeField] private float floatFrequency = 1f;
        [SerializeField] private float rotationSpeed = 10f;
        [SerializeField] private bool randomizeStartTime = true;
        
        private Vector3 originalPosition;
        private float timeOffset;
        
        private void Start()
        {
            originalPosition = transform.localPosition;
            
            if (randomizeStartTime)
            {
                timeOffset = Random.Range(0f, 2f * Mathf.PI);
            }
        }
        
        private void Update()
        {
            // Floating motion
            float newY = originalPosition.y + Mathf.Sin((Time.time + timeOffset) * floatFrequency) * floatAmplitude;
            transform.localPosition = new Vector3(originalPosition.x, newY, originalPosition.z);
            
            // Gentle rotation
            transform.Rotate(Vector3.up, rotationSpeed * Time.deltaTime, Space.Self);
        }
    }
}