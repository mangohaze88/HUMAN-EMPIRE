using UnityEngine;

namespace WorldNavigator.Animation
{
    /// <summary>
    /// Creates pulsing light animation for atmospheric effect
    /// </summary>
    public class LightPulseAnimation : MonoBehaviour
    {
        [Header("Pulse Settings")]
        [SerializeField] private float baseIntensity = 1.5f;
        [SerializeField] private float pulseAmplitude = 0.5f;
        [SerializeField] private float pulseFrequency = 0.8f;
        [SerializeField] private bool randomizePhase = true;
        
        private Light lightComponent;
        private float timeOffset;
        
        private void Start()
        {
            lightComponent = GetComponent<Light>();
            if (lightComponent == null)
            {
                Debug.LogWarning("LightPulseAnimation requires a Light component!");
                enabled = false;
                return;
            }
            
            baseIntensity = lightComponent.intensity;
            
            if (randomizePhase)
            {
                timeOffset = Random.Range(0f, 2f * Mathf.PI);
            }
        }
        
        private void Update()
        {
            if (lightComponent != null)
            {
                float pulse = Mathf.Sin((Time.time + timeOffset) * pulseFrequency) * pulseAmplitude;
                lightComponent.intensity = baseIntensity + pulse;
            }
        }
    }
}