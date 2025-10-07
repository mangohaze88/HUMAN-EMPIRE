using UnityEngine;

namespace WorldNavigator.Core
{
    /// <summary>
    /// Enhanced camera controller for world navigation
    /// </summary>
    public class CameraController : MonoBehaviour
    {
        [Header("Camera Movement")]
        [SerializeField] private float panSpeed = 10f;
        [SerializeField] private float zoomSpeed = 5f;
        [SerializeField] private float rotationSpeed = 100f;
        
        [Header("Limits")]
        [SerializeField] private float minZoom = 10f;
        [SerializeField] private float maxZoom = 100f;
        [SerializeField] private float minHeight = 5f;
        [SerializeField] private float maxHeight = 80f;
        
        [Header("Smoothing")]
        [SerializeField] private float smoothTime = 0.1f;
        
        private Camera cameraComponent;
        private Vector3 targetPosition;
        private Vector3 velocity;
        private Vector3 lastMousePosition;
        private bool isPanning = false;
        
        private void Start()
        {
            cameraComponent = GetComponent<Camera>();
            targetPosition = transform.position;
        }
        
        private void Update()
        {
            HandleInput();
            UpdateMovement();
        }
        
        /// <summary>
        /// Handle camera input
        /// </summary>
        private void HandleInput()
        {
            // Mouse zoom
            float scroll = Input.GetAxis("Mouse ScrollWheel");
            if (Mathf.Abs(scroll) > 0.01f)
            {
                Zoom(-scroll * zoomSpeed);
            }
            
            // Mouse pan (right click)
            if (Input.GetMouseButtonDown(1))
            {
                isPanning = true;
                lastMousePosition = Input.mousePosition;
            }
            else if (Input.GetMouseButtonUp(1))
            {
                isPanning = false;
            }
            
            if (isPanning)
            {
                Vector3 mouseDelta = Input.mousePosition - lastMousePosition;
                Pan(mouseDelta);
                lastMousePosition = Input.mousePosition;
            }
            
            // Keyboard movement
            Vector3 inputVector = Vector3.zero;
            if (Input.GetKey(KeyCode.W) || Input.GetKey(KeyCode.UpArrow))
                inputVector += Vector3.forward;
            if (Input.GetKey(KeyCode.S) || Input.GetKey(KeyCode.DownArrow))
                inputVector += Vector3.back;
            if (Input.GetKey(KeyCode.A) || Input.GetKey(KeyCode.LeftArrow))
                inputVector += Vector3.left;
            if (Input.GetKey(KeyCode.D) || Input.GetKey(KeyCode.RightArrow))
                inputVector += Vector3.right;
            
            if (inputVector.magnitude > 0.1f)
            {
                MoveKeyboard(inputVector.normalized);
            }
            
            // Rotation
            if (Input.GetKey(KeyCode.Q))
                Rotate(-rotationSpeed * Time.deltaTime);
            if (Input.GetKey(KeyCode.E))
                Rotate(rotationSpeed * Time.deltaTime);
        }
        
        /// <summary>
        /// Zoom camera in/out
        /// </summary>
        private void Zoom(float zoomAmount)
        {
            Vector3 forward = transform.forward;
            targetPosition += forward * zoomAmount;
            
            // Clamp height
            targetPosition.y = Mathf.Clamp(targetPosition.y, minHeight, maxHeight);
        }
        
        /// <summary>
        /// Pan camera with mouse
        /// </summary>
        private void Pan(Vector3 mouseDelta)
        {
            Vector3 worldDelta = cameraComponent.ScreenToWorldPoint(new Vector3(mouseDelta.x, mouseDelta.y, transform.position.y));
            Vector3 right = transform.right;
            Vector3 up = transform.up;
            
            targetPosition -= right * mouseDelta.x * panSpeed * 0.01f;
            targetPosition -= up * mouseDelta.y * panSpeed * 0.01f;
        }
        
        /// <summary>
        /// Move camera with keyboard
        /// </summary>
        private void MoveKeyboard(Vector3 direction)
        {
            Vector3 worldDirection = transform.TransformDirection(direction);
            worldDirection.y = 0; // Keep horizontal movement only
            targetPosition += worldDirection * panSpeed * Time.deltaTime;
        }
        
        /// <summary>
        /// Rotate camera around Y axis
        /// </summary>
        private void Rotate(float angle)
        {
            transform.Rotate(Vector3.up, angle, Space.World);
        }
        
        /// <summary>
        /// Update smooth movement
        /// </summary>
        private void UpdateMovement()
        {
            transform.position = Vector3.SmoothDamp(transform.position, targetPosition, ref velocity, smoothTime);
        }
        
        /// <summary>
        /// Focus camera on specific position
        /// </summary>
        public void FocusOn(Vector3 position)
        {
            Vector3 offset = transform.position - transform.forward * 20f;
            targetPosition = position + offset;
            targetPosition.y = Mathf.Clamp(targetPosition.y, minHeight, maxHeight);
        }
        
        /// <summary>
        /// Reset camera to default position
        /// </summary>
        public void ResetView()
        {
            targetPosition = new Vector3(0, 40, -30);
            transform.rotation = Quaternion.Euler(45, 0, 0);
        }
    }
}