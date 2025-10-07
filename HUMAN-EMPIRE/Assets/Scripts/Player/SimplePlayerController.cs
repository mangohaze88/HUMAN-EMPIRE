using UnityEngine;

namespace WorldNavigator.Player
{
    /// <summary>
    /// Simple player character controller
    /// </summary>
    public class SimplePlayerController : MonoBehaviour
    {
        [Header("Movement")]
        [SerializeField] private float moveSpeed = 5f;
        [SerializeField] private float jumpForce = 8f;
        [SerializeField] private float rotationSpeed = 180f;
        
        [Header("Ground Detection")]
        [SerializeField] private LayerMask groundLayer = 1;
        [SerializeField] private float groundCheckDistance = 1.1f;
        
        private Rigidbody rb;
        private bool isGrounded;
        private Vector3 moveDirection;
        
        private void Start()
        {
            SetupPlayer();
        }
        
        /// <summary>
        /// Setup player components
        /// </summary>
        private void SetupPlayer()
        {
            // Add Rigidbody if not present
            rb = GetComponent<Rigidbody>();
            if (rb == null)
            {
                rb = gameObject.AddComponent<Rigidbody>();
            }
            
            rb.freezeRotation = true; // Prevent physics rotation
            
            // Add collider if not present
            if (GetComponent<Collider>() == null)
            {
                CapsuleCollider collider = gameObject.AddComponent<CapsuleCollider>();
                collider.height = 2f;
                collider.radius = 0.5f;
            }
            
            // Create simple player visual
            CreatePlayerVisual();
        }
        
        /// <summary>
        /// Create simple player representation
        /// </summary>
        private void CreatePlayerVisual()
        {
            // Body
            GameObject body = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            body.name = "Player Body";
            body.transform.SetParent(transform);
            body.transform.localPosition = Vector3.zero;
            body.transform.localScale = new Vector3(0.8f, 1f, 0.8f);
            
            // Remove the collider from visual (we have one on parent)
            Destroy(body.GetComponent<Collider>());
            
            // Player material
            Material playerMaterial = new Material(Shader.Find("Standard"));
            playerMaterial.color = new Color(0.2f, 0.6f, 1f, 1f); // Blue
            body.GetComponent<Renderer>().material = playerMaterial;
            
            // Head
            GameObject head = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            head.name = "Player Head";
            head.transform.SetParent(transform);
            head.transform.localPosition = Vector3.up * 1.3f;
            head.transform.localScale = Vector3.one * 0.6f;
            
            Destroy(head.GetComponent<Collider>());
            
            Material headMaterial = new Material(Shader.Find("Standard"));
            headMaterial.color = new Color(1f, 0.8f, 0.6f, 1f); // Skin tone
            head.GetComponent<Renderer>().material = headMaterial;
            
            // Eyes
            CreateEye(new Vector3(-0.15f, 1.4f, 0.4f));
            CreateEye(new Vector3(0.15f, 1.4f, 0.4f));
        }
        
        /// <summary>
        /// Create simple eye
        /// </summary>
        private void CreateEye(Vector3 localPosition)
        {
            GameObject eye = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            eye.name = "Eye";
            eye.transform.SetParent(transform);
            eye.transform.localPosition = localPosition;
            eye.transform.localScale = Vector3.one * 0.1f;
            
            Destroy(eye.GetComponent<Collider>());
            
            Material eyeMaterial = new Material(Shader.Find("Standard"));
            eyeMaterial.color = Color.black;
            eye.GetComponent<Renderer>().material = eyeMaterial;
        }
        
        private void Update()
        {
            HandleInput();
            CheckGrounded();
        }
        
        private void FixedUpdate()
        {
            Move();
        }
        
        /// <summary>
        /// Handle player input
        /// </summary>
        private void HandleInput()
        {
            // Movement input
            float horizontal = Input.GetAxis("Horizontal");
            float vertical = Input.GetAxis("Vertical");
            
            moveDirection = new Vector3(horizontal, 0, vertical).normalized;
            
            // Jump input
            if (Input.GetKeyDown(KeyCode.Space) && isGrounded)
            {
                Jump();
            }
        }
        
        /// <summary>
        /// Check if player is on ground
        /// </summary>
        private void CheckGrounded()
        {
            isGrounded = Physics.Raycast(transform.position, Vector3.down, groundCheckDistance, groundLayer);
        }
        
        /// <summary>
        /// Move player
        /// </summary>
        private void Move()
        {
            if (moveDirection.magnitude > 0.1f)
            {
                // Move
                Vector3 movement = moveDirection * moveSpeed;
                rb.velocity = new Vector3(movement.x, rb.velocity.y, movement.z);
                
                // Rotate towards movement direction
                Quaternion targetRotation = Quaternion.LookRotation(moveDirection);
                transform.rotation = Quaternion.RotateTowards(transform.rotation, targetRotation, rotationSpeed * Time.fixedDeltaTime);
            }
        }
        
        /// <summary>
        /// Make player jump
        /// </summary>
        private void Jump()
        {
            rb.velocity = new Vector3(rb.velocity.x, jumpForce, rb.velocity.z);
        }
        
        /// <summary>
        /// Teleport player to position
        /// </summary>
        public void TeleportTo(Vector3 position)
        {
            transform.position = position + Vector3.up * 2f; // Spawn slightly above
        }
    }
}