using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using OpenCvSharp;
using OpenCvSharp.Extensions;

namespace VideoFrameExtractor
{
    class Program
    {
        static void Main(string[] args)
        {
            // Configuration
            string videoPath = @"C:\Users\USER\Desktop\Python_Deepfake\0924.mp4";
            string outputDirectory = @"C:\Users\USER\Desktop\Python_Deepfake\deepfake_analysis";
            int framesPerClip = 16;
            
            // Check if video file exists
            if (!File.Exists(videoPath))
            {
                Console.WriteLine($"Error: Video file not found at {videoPath}");
                Console.WriteLine("Please make sure:");
                Console.WriteLine("1. The file path is correct");
                Console.WriteLine("2. The video file exists on your desktop");
                Console.WriteLine("3. You've replaced the filename with your actual video file");
                return;
            }
            
            Console.WriteLine("Starting frame extraction and clip creation...");
            Console.WriteLine($"Input video: {videoPath}");
            Console.WriteLine($"Output directory: {outputDirectory}");
            Console.WriteLine($"Frames per clip: {framesPerClip}");
            Console.WriteLine(new string('-', 50));
            
            // Start processing
            ExtractFramesAndCreateClips(videoPath, outputDirectory, framesPerClip);
        }
        
        static void ExtractFramesAndCreateClips(string videoPath, string outputDir, int framesPerClip = 16)
        {
            // Create output directories
            string framesDir = Path.Combine(outputDir, "extracted_frames");
            string clipsDir = Path.Combine(outputDir, "video_clips");
            Directory.CreateDirectory(framesDir);
            Directory.CreateDirectory(clipsDir);
            
            // Open video file
            using var capture = new VideoCapture(videoPath);
            
            if (!capture.IsOpened())
            {
                Console.WriteLine($"Error: Could not open video file {videoPath}");
                return;
            }
            
            // Get video properties
            double fps = capture.Fps;
            int totalFrames = (int)capture.FrameCount;
            int width = (int)capture.FrameWidth;
            int height = (int)capture.FrameHeight;
            
            Console.WriteLine($"Video Info: {totalFrames} frames, {fps:F2} FPS, Resolution: {width}x{height}");
            
            int frameCount = 0;
            int clipCount = 0;
            var framesBuffer = new List<Mat>();
            var frame = new Mat();
            
            while (true)
            {
                capture.Read(frame);
                
                if (frame.Empty())
                    break;
                
                // Save individual frame
                string frameFilename = Path.Combine(framesDir, $"frame_{frameCount:D6}.jpg");
                Cv2.ImWrite(frameFilename, frame);
                
                // Clone frame and add to buffer (we need to clone because Mat is disposed)
                framesBuffer.Add(frame.Clone());
                
                // When we have enough frames for a clip, create the clip
                if (framesBuffer.Count == framesPerClip)
                {
                    CreateVideoClip(framesBuffer, clipsDir, clipCount, fps, width, height);
                    Console.WriteLine($"Created clip {clipCount} with frames {frameCount - framesPerClip + 1} to {frameCount}");
                    
                    // Clear buffer and increment clip count
                    foreach (var bufferedFrame in framesBuffer)
                    {
                        bufferedFrame.Dispose();
                    }
                    framesBuffer.Clear();
                    clipCount++;
                }
                
                frameCount++;
                
                // Progress indicator
                if (frameCount % 100 == 0)
                {
                    Console.WriteLine($"Processed {frameCount}/{totalFrames} frames");
                }
            }
            
            frame.Dispose();
            
            // Create clip from remaining frames (if any)
            if (framesBuffer.Any())
            {
                CreateVideoClip(framesBuffer, clipsDir, clipCount, fps, width, height);
                Console.WriteLine($"Created final clip {clipCount} with {framesBuffer.Count} frames");
                
                foreach (var bufferedFrame in framesBuffer)
                {
                    bufferedFrame.Dispose();
                }
            }
            
            Console.WriteLine($"\nExtraction complete!");
            Console.WriteLine($"Total frames extracted: {frameCount}");
            Console.WriteLine($"Total clips created: {clipCount + (framesBuffer.Any() ? 1 : 0)}");
            Console.WriteLine($"Frames saved in: {framesDir}");
            Console.WriteLine($"Clips saved in: {clipsDir}");
        }
        
        static void CreateVideoClip(List<Mat> frames, string outputDir, int clipNumber, double fps, int width, int height)
        {
            string clipFilename = Path.Combine(outputDir, $"clip_{clipNumber:D4}.avi");
            
            using var writer = new VideoWriter(clipFilename, FourCC.XVID, fps, new Size(width, height));
            
            foreach (var frame in frames)
            {
                writer.Write(frame);
            }
        }
    }
}