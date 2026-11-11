<h1 align="center">
  Auto Maple
</h1>

Auto Maple is an intelligent Python AI that plays MapleStory, a 2D side-scrolling MMORPG, using simulated key presses, TensorFlow machine learning, OpenCV template matching, and other computer vision techniques.

Community-created resources, such as **command books** for each class and **routines** for each map, can be found in the **[resources repository](https://github.com/tanjeffreyz/auto-maple-resources)**.

<br>


<h2 align="center">
  Minimap
</h2>

<table align="center" border="0">
  <tr>
    <td>
Auto Maple uses <b>OpenCV template matching</b> to determine the bounds of the minimap as well as the various elements within it, allowing it to accurately track the player's in-game position. If <code>record_layout</code> is set to <code>True</code>, Auto Maple will record the player's previous positions in a <b>quadtree-based</b> Layout object, which is periodically saved to a file in the "layouts" directory. Every time a new routine is loaded, its corresponding layout file, if it exists, will also be loaded. This Layout object uses the <b>A* search algorithm</b> on its stored points to calculate the shortest path from the player to any target location, which can dramatically improve the accuracy and speed at which routines are executed.
    </td>
    <td align="center" width="400px">
      <img align="center" src="https://user-images.githubusercontent.com/69165598/123177212-b16f0700-d439-11eb-8a21-8b414273f1e1.gif"/>
    </td>
  </tr>
</table>

<br>








<h2 align="center">
  Command Books
</h2>

<p align="center">
  <img src="https://user-images.githubusercontent.com/69165598/123372905-502e5d00-d539-11eb-81c2-46b8bbf929cc.gif" width="100%"/>
  <br>
  <sub>
    The above video shows Auto Maple consistently performing a mechanically advanced ability combination.
  </sub>
</p>
  
<table align="center" border="0">
  <tr>
    <td width="100%">
Designed with modularity in mind, Auto Maple can operate any character in the game as long as it is provided with a list of in-game actions, or a "command book". A command book is a Python file that contains multiple classes, one for each in-game ability, that tells the program what keys it should press and when to press them. Once a command book is imported, its classes are automatically compiled into a dictionary that Auto Maple can then use to interpret commands within routines. Commands have access to all of Auto Maple's global variables, which can allow them to actively change their behavior based on the player's position and the state of the game.
    </td>
  </tr>
</table>
  
<br>







<h2 align="center">
  Routines
</h2>

<table align="center" border="0">
  <tr>
    <td width="350px">
      <p align="center">
        <img src="https://user-images.githubusercontent.com/69165598/150469699-d8a94ab4-7d70-49c3-8736-a9018996f39a.png"/>
        <br>
        <sub>
          Click <a href="https://github.com/tanjeffreyz02/auto-maple/blob/f13d87c98e9344e0a4fa5c6f85ffb7e66860afc0/routines/dcup2.csv">here</a> to view the entire routine.
        </sub>
      </p>
    </td>
    <td>
A routine is a user-created CSV file that tells Auto Maple where to move and what commands to use at each location. A custom compiler within Auto Maple parses through the selected routine and converts it into a list of <code>Component</code> objects that can then be executed by the program. An error message is printed for every line that contains invalid parameters, and those lines are ignored during the conversion. 
<br><br>
Below is a summary of the most commonly used routine components:
<ul>
  <li>
    <b><code>Point</code></b> stores the commands directly below it and will execute them in that order once the character is within <code>move_tolerance</code> of the specified location. There are also a couple optional keyword arguments:
    <ul>
      <li>
        <code>adjust</code> fine-tunes the character's position to be within <code>adjust_tolerance</code> of the target location before executing any commands.
      </li>
      <li>
        <code>frequency</code> tells the Point how often to execute. If set to N, this Point will execute once every N iterations.
      </li>
      <li>
        <code>skip</code> tells the Point whether to run on the first iteration or not. If set to True and frequency is N, this Point will execute on the N-1th iteration.
      </li>
    </ul>
  </li>
  <li>
    <b><code>Label</code></b> acts as a reference point that can help organize the routine into sections as well as create loops.
  </li>
  <li>
    <b><code>Jump</code></b> jumps to the given label from anywhere in the routine.
  </li>
  <li>
    <b><code>Setting</code></b> updates the specified setting to the given value. It can be placed anywhere in the routine, so different parts of the same routine can have different settings. All editable settings can be found at the bottom of <a href="https://github.com/tanjeffreyz02/auto-maple/blob/v2/settings.py">settings.py</a>.
  </li>
</ul>
    </td>
  </tr>
</table>

<br>








<h2 align="center">
  Runes
</h2>

<p align="center">
  <img src="https://user-images.githubusercontent.com/69165598/123479558-f61fad00-d5b5-11eb-914c-8f002a96dd62.gif" width="100%"/>
</p>

<table align="center" border="0">
  <tr>
    <td width="100%">
Auto Maple has the ability to automatically solve "runes", or in-game arrow key puzzles. It first uses OpenCV's color filtration and <b>Canny edge detection</b> algorithms to isolate the arrow keys and reduce as much background noise as possible. Then, it runs multiple inferences on the preprocessed frames using a custom-trained <b>TensorFlow</b> model until two inferences agree. Because of this preprocessing, Auto Maple is extremely accurate at solving runes in all kinds of (often colorful and chaotic) environments.
    </td>
  </tr>
</table>

<h2>Experimental shape challenge tracker</h2>

<p>The experimental tracker simplifies the initial silhouette to a stable polygon, then follows feature points sampled along its boundary. Forward/backward optical flow and a RANSAC affine fit estimate movement from surviving edge points. It does not treat the animated, swirling background as a single motion reference; when target evidence is weak, it continues a damped trajectory prediction, even if the predicted outline leaves the video frame. The live cursor is clamped separately to the screen. The displayed box includes the saved outline and its tracked points. Accepted matches are green; a yellow box, cyan outline, and blue edge points show prediction when a match is unavailable. Prediction is not a confirmed match: if the target has fully disappeared among similar shapes, the tracker cannot know its identity from that frame alone.</p>

<p>Preview a recording with the confidence-gated silhouette tracker before trying live cursor following:</p>

<pre><code>python -m src.detection.shape_challenge --video "path\to\recording.mp4" --output "tracked.mp4" --metrics "tracking.csv"</code></pre>

<p>To create persistent point labels for optical-flow evaluation, mark the target center first, followed by at least two distinctive features in the same order on every sampled frame. Use a shorter interval around fast movement or fading; skip frames where the target cannot be identified confidently. This produces manual reference tracks, not an automatic recovery of a fully invisible target.</p>

<pre><code>python -m src.detection.shape_polygon_labeler --video "path\to\recording.mp4" --mode points --interval 0.25 --output "assets\shape_training\shape_point_labels.json"</code></pre>

<p>For live use, start in preview mode with <code>python -m src.detection.shape_challenge</code>. Cursor movement is opt-in with <code>--move-cursor</code>; the tracker does not click or send keys. When tracking is lost, cursor following is limited to a short, confidence-gated prediction (up to six frames), then pauses until visual tracking returns. The CSV records each frame's estimate and confidence. Cursor reference measurements are available only when the recording exposes the green cursor marker, and are not ground-truth target error. Accuracy depends on the recording and has not been validated to a 90% success rate.</p>


<br>









<h2 align="center">
  Video Demonstration
</h2>

<p align="center">
  <a href="https://youtu.be/iNj1CWW2--8?si=MA4n6EAHokI9FX8B"><b>Click below to watch the full video</b></a>
</p>

<p align="center">
  <a href="https://youtu.be/iNj1CWW2--8?si=MA4n6EAHokI9FX8B">
    <img src="https://user-images.githubusercontent.com/69165598/123308656-c5b61100-d4d8-11eb-99ac-c465665474b5.gif" width="600px"/>
  </a>
</p>

<br>



<h2 align="center">
  Setup
</h2>

<ol>
  <li>
    Download and install <a href="https://www.python.org/downloads/">Python3</a>.
  </li>
  <li>
    Download and install the latest version of <a href="https://developer.nvidia.com/cuda-downloads">CUDA Toolkit</a>.
  </li>
  <li>
    Download and install <a href="https://git-scm.com/download/win">Git</a>.
  </li>
  <li>
    Download and unzip the latest <a href="https://github.com/tanjeffreyz02/auto-maple/releases">Auto Maple release</a>.
  </li>
  <li>
    Download the <a href="https://drive.google.com/drive/folders/1SPdTNF4KZczoWyWTgfyTBRvLvy7WSGpu?usp=sharing">TensorFlow model</a> and unzip the "models" folder into Auto Maple's "assets" directory. Without a complete model, Auto Maple can start but will pause its routine when a rune is detected; handle the puzzle manually and resume with the Start/stop key. Keep the in-game minimap visible at its default size so Auto Maple can locate its corners. If automatic minimap calibration fails, drag around the minimap map area in the selection dialog (include the player marker but exclude the title bar) and confirm. Press F6 to request manual selection again. The GUI opens after the first screenshot even if minimap calibration is still retrying; movement remains unavailable until calibration succeeds.
  </li>
  <li>
    Inside Auto Maple's main directory, open a command prompt and run:
    <pre><code>python -m pip install -r requirements.txt</code></pre>
  </li>
  <li>
    Lastly, create a desktop shortcut by running:
    <pre><code>python setup.py</code></pre>
    This shortcut uses absolute paths, so feel free to move it wherever you want. However, if you move Auto Maple's main directory, you will need to run <code>python setup.py</code> again to generate a new shortcut. To keep the command prompt open after Auto Maple closes, run the above command with the <code>--stay</code> flag.
  </li>
</ol>

<h2>Basic routine</h2>

<p>For the included <code>begginer.py</code> command book, bind left/right movement to <code>A</code>/<code>D</code>, jump to <code>Space</code>, and Cruel Stab to <code>Ctrl</code> in-game. In Auto Maple, choose <b>File → Load Command Book</b> and select <code>begginer.py</code>. After minimap calibration succeeds, press <code>F7</code> at the first position, then click the recorded position in the Edit tab to create a Point. Select that Point, use <code>+</code> to add <code>CruelStabRandomDirection</code>, and confirm it with Add. Repeat at a second position and save the routine. Press <code>Insert</code> to start and again to pause. The routine visits the Points in order and repeats from the first Point; test while paused/ready to stop and supervise it.</p>

<p>Optional HP potion use is configured in the View tab under Status. The monitor locates the red HP fill and its bar track near the bottom of the captured game window, so window position and size do not require manual bar calibration. Set the potion key (default <code>R</code>) and enable Auto HP potion. While the option is enabled and MapleStory is focused, it presses the potion key every 1.5 seconds while detected HP is at or below 50%, stopping once HP rises above 50%; this remains active even when Auto Maple is paused. A changed HUD layout or overlays can affect detection; verify the displayed HP before relying on it.</p>

<p>When a routine is running, its Move commands temporarily pause the patrol to approach a detected enemy horizontally if it is within the tracker’s calibrated vertical band. It attacks while the target remains locked, then resumes the interrupted waypoint after the target is no longer detected. This is not platform navigation: it will not jump or drop to another level, and the vertical band is only an approximate same-level filter. Supervise the first run.</p>

<p>Sampled-enemy auto-attack is available in the View tab under Status. Choose <b>Calibrate player + enemies</b> and select the player name badge in the play area (not the duplicate name in the HUD). On the next image, drag a tight box around each complete enemy sprite you want to sample, including each visually distinct color/type (for example, different snail shell colors), excluding nearby drops and background, then choose <b>Finish</b>. Samples retain color as well as shape; recalibrate existing grayscale samples once to enable color matching. Matching checks each sample at several nearby scales (40–120%) to tolerate size changes. All selected samples are saved together from that same screenshot. Use <b>Add enemy sample</b> later to append another sample. Enable auto-attack and load a command book that provides <code>CruelStab</code>. With no routine loaded, press the configured Start/stop key (default <code>Insert</code>), focus MapleStory during the 3-second countdown, and Auto Maple will sweep left/right, reversing every few seconds, confirm and lock the nearest sampled enemy, then send two attacks about every 0.4 seconds while it remains visible and within range. It keeps the lock and approaches/attacks through brief sprite/animation match loss; after about 1.5 seconds without a match, it releases that target and confirms the next closest sample. Movement stops and attacks begin within 60 screen pixels of the target center to reduce overshooting as the player closes in. Target choice is based on distance, not enemy level or maximum HP; a green HP bar is shown as a rough remaining-HP estimate, not used to drop a live target. Search range is adjustable. Auto Maple also uses a simple group heuristic: three or more sampled enemies detected close together on the calibrated level trigger a brief turn toward the group and one <code>Q</code> tap per attack cooldown while the group remains in range; otherwise, the loaded <code>CruelStab</code> command is used. Detection allows for enemies jumping up to about 64 screen pixels above or below their calibrated level; single-target attacks still require the tighter 16-pixel level alignment, while the group skill may trigger within the wider jump band. Bind the area skill to <code>Q</code> in-game before enabling this behavior. The group count uses template matches, not a trained detector, so animation, overlapping enemies, effects, and scale changes can cause missed or false groups; supervise and test it before relying on it. Auto Maple does not identify platform edges or descend to the bottom floor automatically. Start on the level you want to patrol and supervise movement. Template matching can miss or falsely match enemies because of animation, visual effects, UI changes, or a different game scale; enemy types not included in the samples will not be targeted. Re-sample enemies on the level and at the same game scale you plan to use if the tracker does not detect them.</p>

<p>Optional loot pickup is configured in the same Status area. First calibrate the player badge with <b>Calibrate player + enemies</b>; loot pickup reuses that badge but needs no loot calibration for common drops. Built-in samples cover the coin, purple, and white drops shown in the example, and matching checks several item scales and follows small vertical bobbing between scans. Use <b>Add map-specific loot sample</b> only for unusual drops or maps; tightly box each complete icon, excluding ground, nearby items, and effects. While auto-pickup is checked, detection status is visible even with Auto Maple paused, but no keys are sent until Auto Maple is running and MapleStory is focused. Pickup does not move toward loot or change the mob-search route. It requests six quick <code>X</code> taps for a matching drop directly under the player; low-confidence matches must also show vertical movement across scans. One burst is sent per detected item, and the detector rearms after the item is no longer seen. The status shows match scores when nothing matches. Item animation, overlap, UI changes, and platform changes can affect matching; supervise it and start by testing one item.</p>

<p>Runtime messages and errors are also available in the <b>Log</b> tab. Notifications for minimap/player events and safety pauses are silent; their messages appear in the console and Log tab. The input overlay reports keys requested through <code>SendInput</code>, including mapped movement keys; this confirms only that Auto Maple issued the request, not that the game received it. Output continues to appear in the console window, and the Log tab keeps a bounded recent history.</p>
