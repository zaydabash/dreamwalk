using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using WebSocketSharp;
using Newtonsoft.Json;

namespace DreamWalk
{
    /// <summary>
    /// Main controller for DreamWalk VR world generation and state management
    /// </summary>
    public class DreamController : MonoBehaviour
    {
        [Header("WebSocket Configuration")]
        [SerializeField] private string websocketUrl = "ws://localhost:8004/ws/";
        [SerializeField] private string sessionId = "unity_demo";
        [SerializeField] private bool autoConnect = true;
        [SerializeField] private float reconnectInterval = 5f;
        
        [Header("World Generation")]
        [SerializeField] private Transform worldParent;
        [SerializeField] private GameObject[] biomePrefabs;
        [SerializeField] private Material[] skyboxMaterials;
        [SerializeField] private Gradient[] lightingGradients;
        
        [Header("Audio")]
        [SerializeField] private AudioSource ambientAudioSource;
        [SerializeField] private AudioSource musicAudioSource;
        [SerializeField] private AudioClip[] ambientClips;
        [SerializeField] private AudioClip[] musicClips;
        
        [Header("Visual Effects")]
        [SerializeField] private Volume postProcessVolume;
        [SerializeField] private ParticleSystem[] particleSystems;
        [SerializeField] private Light mainLight;
        [SerializeField] private Light[] accentLights;
        
        [Header("Animation Curves")]
        [SerializeField] private AnimationCurve weatherIntensityCurve = AnimationCurve.Linear(0, 0, 1, 1);
        [SerializeField] private AnimationCurve objectDensityCurve = AnimationCurve.Linear(0, 0, 1, 1);
        [SerializeField] private AnimationCurve morphSpeedCurve = AnimationCurve.Linear(0, 0.5f, 1, 2f);
        
        [Header("Debug")]
        [SerializeField] private bool enableDebugMode = true;
        [SerializeField] private KeyCode debugToggleKey = KeyCode.F1;
        [SerializeField] private KeyCode manualStateKey = KeyCode.F2;
        
        // Private fields
        private WebSocket webSocket;
        private bool isConnected = false;
        private Coroutine reconnectCoroutine;
        private WorldState currentWorldState;
        private WorldState targetWorldState;
        private bool isMorphing = false;
        private float morphProgress = 0f;
        
        // Component references
        private VolumeProfile postProcessProfile;
        private ColorAdjustments colorAdjustments;
        private Fog fog;
        private Vignette vignette;
        
        // Biome management
        private Dictionary<string, GameObject> activeBiomes = new Dictionary<string, GameObject>();
        private string currentBiomeType = "neutral";
        
        // Audio management
        private Dictionary<string, AudioClip> ambientClipMap;
        private Dictionary<string, AudioClip> musicClipMap;
        private Coroutine audioTransitionCoroutine;
        
        // Events
        public event Action<WorldState> OnWorldStateUpdated;
        public event Action<string> OnBiomeChanged;
        public event Action<float> OnMorphProgress;
        
        private void Start()
        {
            InitializeComponents();
            SetupPostProcessing();
            InitializeAudio();
            
            if (autoConnect)
            {
                ConnectToServer();
            }
        }
        
        private void Update()
        {
            HandleInput();
            UpdateWorldMorphing();
        }
        
        private void OnDestroy()
        {
            DisconnectFromServer();
        }
        
        #region Initialization
        
        private void InitializeComponents()
        {
            // Get post-processing components
            if (postProcessVolume != null)
            {
                postProcessProfile = postProcessVolume.profile;
                if (postProcessProfile.TryGet<ColorAdjustments>(out colorAdjustments) == false)
                {
                    colorAdjustments = postProcessProfile.Add<ColorAdjustments>();
                }
                if (postProcessProfile.TryGet<Fog>(out fog) == false)
                {
                    fog = postProcessProfile.Add<Fog>();
                }
                if (postProcessProfile.TryGet<Vignette>(out vignette) == false)
                {
                    vignette = postProcessProfile.Add<Vignette>();
                }
            }
            
            // Initialize audio clip mappings
            ambientClipMap = new Dictionary<string, AudioClip>();
            musicClipMap = new Dictionary<string, AudioClip>();
            
            for (int i = 0; i < ambientClips.Length; i++)
            {
                ambientClipMap[ambientClips[i].name] = ambientClips[i];
            }
            
            for (int i = 0; i < musicClips.Length; i++)
            {
                musicClipMap[musicClips[i].name] = musicClips[i];
            }
        }
        
        private void SetupPostProcessing()
        {
            // Configure initial post-processing settings
            if (colorAdjustments != null)
            {
                colorAdjustments.colorFilter.value = Color.white;
                colorAdjustments.contrast.value = 0f;
                colorAdjustments.saturation.value = 0f;
            }
            
            if (fog != null)
            {
                fog.enabled.value = true;
                fog.color.value = new Color(0.5f, 0.5f, 0.5f, 1f);
                fog.density.value = 0.1f;
            }
            
            if (vignette != null)
            {
                vignette.enabled.value = false;
            }
        }
        
        private void InitializeAudio()
        {
            if (ambientAudioSource != null)
            {
                ambientAudioSource.loop = true;
                ambientAudioSource.volume = 0.5f;
            }
            
            if (musicAudioSource != null)
            {
                musicAudioSource.loop = true;
                musicAudioSource.volume = 0.3f;
            }
        }
        
        #endregion
        
        #region WebSocket Connection
        
        public void ConnectToServer()
        {
            if (isConnected) return;
            
            try
            {
                string fullUrl = websocketUrl + sessionId;
                webSocket = new WebSocket(fullUrl);
                
                webSocket.OnOpen += OnWebSocketOpen;
                webSocket.OnMessage += OnWebSocketMessage;
                webSocket.OnError += OnWebSocketError;
                webSocket.OnClose += OnWebSocketClose;
                
                webSocket.Connect();
                
                Debug.Log($"Connecting to WebSocket: {fullUrl}");
            }
            catch (Exception e)
            {
                Debug.LogError($"Failed to connect to WebSocket: {e.Message}");
                StartReconnectCoroutine();
            }
        }
        
        public void DisconnectFromServer()
        {
            if (reconnectCoroutine != null)
            {
                StopCoroutine(reconnectCoroutine);
                reconnectCoroutine = null;
            }
            
            if (webSocket != null && webSocket.ReadyState == WebSocketState.Open)
            {
                webSocket.Close();
            }
            
            isConnected = false;
        }
        
        private void StartReconnectCoroutine()
        {
            if (reconnectCoroutine != null)
            {
                StopCoroutine(reconnectCoroutine);
            }
            
            reconnectCoroutine = StartCoroutine(ReconnectCoroutine());
        }
        
        private IEnumerator ReconnectCoroutine()
        {
            while (!isConnected)
            {
                yield return new WaitForSeconds(reconnectInterval);
                
                if (!isConnected)
                {
                    Debug.Log("Attempting to reconnect...");
                    ConnectToServer();
                }
            }
        }
        
        #endregion
        
        #region WebSocket Event Handlers
        
        private void OnWebSocketOpen(object sender, EventArgs e)
        {
            isConnected = true;
            Debug.Log("WebSocket connected successfully");
            
            if (reconnectCoroutine != null)
            {
                StopCoroutine(reconnectCoroutine);
                reconnectCoroutine = null;
            }
        }
        
        private void OnWebSocketMessage(object sender, MessageEventArgs e)
        {
            try
            {
                var message = JsonConvert.DeserializeObject<WebSocketMessage>(e.Data);
                
                switch (message.type)
                {
                    case "connection_established":
                        Debug.Log("Connection established with server");
                        break;
                        
                    case "world_state_update":
                        HandleWorldStateUpdate(message.world_state);
                        break;
                        
                    case "error":
                        Debug.LogError($"Server error: {message.message}");
                        break;
                        
                    case "pong":
                        // Handle ping-pong for connection health
                        break;
                        
                    default:
                        Debug.LogWarning($"Unknown message type: {message.type}");
                        break;
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"Failed to parse WebSocket message: {ex.Message}");
            }
        }
        
        private void OnWebSocketError(object sender, ErrorEventArgs e)
        {
            Debug.LogError($"WebSocket error: {e.Message}");
            isConnected = false;
            StartReconnectCoroutine();
        }
        
        private void OnWebSocketClose(object sender, CloseEventArgs e)
        {
            Debug.Log($"WebSocket closed: {e.Reason}");
            isConnected = false;
            StartReconnectCoroutine();
        }
        
        #endregion
        
        #region World State Management
        
        private void HandleWorldStateUpdate(WorldStateData worldStateData)
        {
            try
            {
                var newWorldState = new WorldState
                {
                    biomeType = worldStateData.biome_type ?? "neutral",
                    weatherIntensity = worldStateData.weather_intensity,
                    lightingMood = worldStateData.lighting_mood ?? "neutral",
                    colorPalette = worldStateData.color_palette ?? new List<float> { 0.5f, 0.5f, 0.5f },
                    objectDensity = worldStateData.object_density,
                    structureLevel = worldStateData.structure_level,
                    ambientVolume = worldStateData.ambient_volume,
                    musicIntensity = worldStateData.music_intensity,
                    soundEffects = worldStateData.sound_effects ?? new List<string>(),
                    changeRate = worldStateData.change_rate,
                    morphSpeed = worldStateData.morph_speed
                };
                
                targetWorldState = newWorldState;
                
                if (!isMorphing)
                {
                    StartCoroutine(MorphToNewState());
                }
                
                OnWorldStateUpdated?.Invoke(newWorldState);
                
                if (enableDebugMode)
                {
                    Debug.Log($"World state updated: {newWorldState.biomeType}");
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"Failed to handle world state update: {e.Message}");
            }
        }
        
        private IEnumerator MorphToNewState()
        {
            isMorphing = true;
            morphProgress = 0f;
            
            var startState = currentWorldState ?? GetDefaultWorldState();
            var duration = 1f / targetWorldState.morphSpeed;
            
            while (morphProgress < 1f)
            {
                morphProgress += Time.deltaTime / duration;
                morphProgress = Mathf.Clamp01(morphProgress);
                
                var lerpedState = LerpWorldStates(startState, targetWorldState, morphProgress);
                ApplyWorldState(lerpedState);
                
                OnMorphProgress?.Invoke(morphProgress);
                
                yield return null;
            }
            
            currentWorldState = targetWorldState;
            isMorphing = false;
        }
        
        private WorldState GetDefaultWorldState()
        {
            return new WorldState
            {
                biomeType = "neutral",
                weatherIntensity = 0.5f,
                lightingMood = "neutral",
                colorPalette = new List<float> { 0.5f, 0.5f, 0.5f },
                objectDensity = 0.5f,
                structureLevel = 0.5f,
                ambientVolume = 0.5f,
                musicIntensity = 0.3f,
                soundEffects = new List<string>(),
                changeRate = 0.1f,
                morphSpeed = 1f
            };
        }
        
        private WorldState LerpWorldStates(WorldState start, WorldState end, float t)
        {
            return new WorldState
            {
                biomeType = end.biomeType, // Don't lerp biome type
                weatherIntensity = Mathf.Lerp(start.weatherIntensity, end.weatherIntensity, t),
                lightingMood = end.lightingMood, // Don't lerp lighting mood
                colorPalette = new List<float>
                {
                    Mathf.Lerp(start.colorPalette[0], end.colorPalette[0], t),
                    Mathf.Lerp(start.colorPalette[1], end.colorPalette[1], t),
                    Mathf.Lerp(start.colorPalette[2], end.colorPalette[2], t)
                },
                objectDensity = Mathf.Lerp(start.objectDensity, end.objectDensity, t),
                structureLevel = Mathf.Lerp(start.structureLevel, end.structureLevel, t),
                ambientVolume = Mathf.Lerp(start.ambientVolume, end.ambientVolume, t),
                musicIntensity = Mathf.Lerp(start.musicIntensity, end.musicIntensity, t),
                soundEffects = end.soundEffects, // Don't lerp sound effects
                changeRate = Mathf.Lerp(start.changeRate, end.changeRate, t),
                morphSpeed = Mathf.Lerp(start.morphSpeed, end.morphSpeed, t)
            };
        }
        
        #endregion
        
        #region World State Application
        
        private void ApplyWorldState(WorldState worldState)
        {
            ApplyBiome(worldState.biomeType);
            ApplyWeather(worldState.weatherIntensity);
            ApplyLighting(worldState.lightingMood, worldState.colorPalette);
            ApplyPostProcessing(worldState);
            ApplyAudio(worldState);
            ApplyParticleEffects(worldState);
        }
        
        private void ApplyBiome(string biomeType)
        {
            if (biomeType != currentBiomeType)
            {
                // Deactivate current biome
                if (activeBiomes.ContainsKey(currentBiomeType))
                {
                    activeBiomes[currentBiomeType].SetActive(false);
                }
                
                // Activate new biome
                if (activeBiomes.ContainsKey(biomeType))
                {
                    activeBiomes[biomeType].SetActive(true);
                }
                else
                {
                    // Instantiate new biome if not exists
                    var biomePrefab = GetBiomePrefab(biomeType);
                    if (biomePrefab != null)
                    {
                        var biomeInstance = Instantiate(biomePrefab, worldParent);
                        biomeInstance.name = $"Biome_{biomeType}";
                        activeBiomes[biomeType] = biomeInstance;
                    }
                }
                
                currentBiomeType = biomeType;
                OnBiomeChanged?.Invoke(biomeType);
                
                if (enableDebugMode)
                {
                    Debug.Log($"Biome changed to: {biomeType}");
                }
            }
        }
        
        private GameObject GetBiomePrefab(string biomeType)
        {
            foreach (var prefab in biomePrefabs)
            {
                if (prefab.name.ToLower().Contains(biomeType.ToLower()))
                {
                    return prefab;
                }
            }
            return biomePrefabs.Length > 0 ? biomePrefabs[0] : null; // Return first prefab as default
        }
        
        private void ApplyWeather(float intensity)
        {
            // Apply fog based on weather intensity
            if (fog != null)
            {
                fog.density.value = Mathf.Lerp(0.05f, 0.3f, intensity);
                
                // Change fog color based on intensity
                var fogColor = Color.Lerp(new Color(0.7f, 0.7f, 0.8f), new Color(0.3f, 0.3f, 0.4f), intensity);
                fog.color.value = fogColor;
            }
            
            // Apply vignette for stormy weather
            if (vignette != null)
            {
                vignette.enabled.value = intensity > 0.7f;
                vignette.intensity.value = (intensity - 0.7f) * 3.33f; // Scale to 0-1
            }
        }
        
        private void ApplyLighting(string mood, List<float> colorPalette)
        {
            // Apply main light color
            if (mainLight != null && colorPalette.Count >= 3)
            {
                var lightColor = new Color(colorPalette[0], colorPalette[1], colorPalette[2]);
                mainLight.color = lightColor;
                
                // Adjust intensity based on mood
                switch (mood)
                {
                    case "bright_warm":
                        mainLight.intensity = 1.5f;
                        mainLight.colorTemperature = 3000f;
                        break;
                    case "dim_cool":
                        mainLight.intensity = 0.5f;
                        mainLight.colorTemperature = 7000f;
                        break;
                    default:
                        mainLight.intensity = 1f;
                        mainLight.colorTemperature = 5000f;
                        break;
                }
            }
            
            // Apply skybox
            var skyboxMaterial = GetSkyboxMaterial(mood);
            if (skyboxMaterial != null)
            {
                RenderSettings.skybox = skyboxMaterial;
            }
        }
        
        private Material GetSkyboxMaterial(string mood)
        {
            foreach (var material in skyboxMaterials)
            {
                if (material.name.ToLower().Contains(mood.ToLower()))
                {
                    return material;
                }
            }
            return skyboxMaterials.Length > 0 ? skyboxMaterials[0] : null;
        }
        
        private void ApplyPostProcessing(WorldState worldState)
        {
            // Apply color adjustments
            if (colorAdjustments != null && worldState.colorPalette.Count >= 3)
            {
                var colorFilter = new Color(worldState.colorPalette[0], worldState.colorPalette[1], worldState.colorPalette[2]);
                colorAdjustments.colorFilter.value = colorFilter;
                
                // Adjust contrast and saturation based on structure level
                colorAdjustments.contrast.value = (worldState.structureLevel - 0.5f) * 20f;
                colorAdjustments.saturation.value = (worldState.structureLevel - 0.5f) * 50f;
            }
        }
        
        private void ApplyAudio(WorldState worldState)
        {
            // Update ambient audio volume
            if (ambientAudioSource != null)
            {
                ambientAudioSource.volume = worldState.ambientVolume;
            }
            
            // Update music volume
            if (musicAudioSource != null)
            {
                musicAudioSource.volume = worldState.musicIntensity;
            }
            
            // Handle sound effects
            if (worldState.soundEffects.Count > 0)
            {
                PlaySoundEffects(worldState.soundEffects);
            }
        }
        
        private void PlaySoundEffects(List<string> soundEffects)
        {
            // Implementation for playing sound effects
            // This would involve finding appropriate AudioClips and playing them
            foreach (var effect in soundEffects)
            {
                if (enableDebugMode)
                {
                    Debug.Log($"Playing sound effect: {effect}");
                }
            }
        }
        
        private void ApplyParticleEffects(WorldState worldState)
        {
            foreach (var particleSystem in particleSystems)
            {
                if (particleSystem != null)
                {
                    var emission = particleSystem.emission;
                    emission.rateOverTime = worldState.objectDensity * 100f;
                    
                    // Adjust particle properties based on world state
                    var main = particleSystem.main;
                    main.startColor = new Color(worldState.colorPalette[0], worldState.colorPalette[1], worldState.colorPalette[2], 0.5f);
                }
            }
        }
        
        #endregion
        
        #region Input Handling
        
        private void HandleInput()
        {
            if (Input.GetKeyDown(debugToggleKey))
            {
                enableDebugMode = !enableDebugMode;
                Debug.Log($"Debug mode: {enableDebugMode}");
            }
            
            if (Input.GetKeyDown(manualStateKey))
            {
                TriggerManualWorldUpdate();
            }
        }
        
        private void TriggerManualWorldUpdate()
        {
            if (isConnected && webSocket != null)
            {
                var message = new
                {
                    type = "request_world_update",
                    timestamp = DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
                };
                
                webSocket.Send(JsonConvert.SerializeObject(message));
            }
        }
        
        #endregion
        
        #region Utility Methods
        
        private void UpdateWorldMorphing()
        {
            // Update morphing progress for smooth transitions
            if (isMorphing)
            {
                // Additional morphing logic can be added here
            }
        }
        
        public void SetManualWorldState(WorldState worldState)
        {
            if (isConnected && webSocket != null)
            {
                var message = new
                {
                    type = "set_manual_state",
                    world_state = worldState,
                    timestamp = DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
                };
                
                webSocket.Send(JsonConvert.SerializeObject(message));
            }
        }
        
        public bool IsConnected()
        {
            return isConnected;
        }
        
        public WorldState GetCurrentWorldState()
        {
            return currentWorldState;
        }
        
        #endregion
    }
    
    #region Data Classes
    
    [Serializable]
    public class WorldState
    {
        public string biomeType;
        public float weatherIntensity;
        public string lightingMood;
        public List<float> colorPalette;
        public float objectDensity;
        public float structureLevel;
        public float ambientVolume;
        public float musicIntensity;
        public List<string> soundEffects;
        public float changeRate;
        public float morphSpeed;
    }
    
    [Serializable]
    public class WorldStateData
    {
        public string biome_type;
        public float weather_intensity;
        public string lighting_mood;
        public List<float> color_palette;
        public float object_density;
        public float structure_level;
        public float ambient_volume;
        public float music_intensity;
        public List<string> sound_effects;
        public float change_rate;
        public float morph_speed;
    }
    
    [Serializable]
    public class WebSocketMessage
    {
        public string type;
        public string session_id;
        public string timestamp;
        public string message;
        public WorldStateData world_state;
    }
    
    #endregion
}
