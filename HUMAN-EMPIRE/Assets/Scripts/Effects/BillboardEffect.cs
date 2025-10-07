using UnityEngine;

namespace WorldNavigator.Effects
{
    /// <summary>
    /// Makes UI elements always face the camera
    /// </summary>
    public class BillboardEffect : MonoBehaviour
    {
        [Header("Billboard Settings")]
        [SerializeField] private bool lockY = true;
        [SerializeField] private bool lockX = false;
        [SerializeField] private bool lockZ = false;
        
        private Camera targetCamera;
        
        private void Start()
        {
            targetCamera = Camera.main;
            if (targetCamera == null)
            {
                targetCamera = FindFirstObjectByType<Camera>();
            }
        }
        
        private void LateUpdate()
        {
            if (targetCamera == null) return;
            
            Vector3 directionToCamera = targetCamera.transform.position - transform.position;
            
            if (lockY) directionToCamera.y = 0;
            if (lockX) directionToCamera.x = 0;
            if (lockZ) directionToCamera.z = 0;
            
            if (directionToCamera != Vector3.zero)
            {
                transform.rotation = Quaternion.LookRotation(-directionToCamera);
            }
        }
    }
}