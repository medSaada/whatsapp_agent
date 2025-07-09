#!/usr/bin/env python3
"""
Test script for WhatsApp Media Upload functionality

This script demonstrates how to use the media upload functionality programmatically.
Run this to test if your media upload is working correctly.
"""

import os
import sys

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

try:
    from app.services.meta_api_client import MetaAPIClient
    from app.core.config import get_settings
    from app.core.logging import get_logger
except ImportError:
    print("Error: Could not import app modules. Make sure you're running this from the project root.")
    sys.exit(1)

logger = get_logger()

def test_media_upload():
    """Test the media upload functionality"""
    print("🧪 Testing WhatsApp Media Upload")
    print("=" * 50)
    
    try:
        # Initialize the API client
        settings = get_settings()
        client = MetaAPIClient(settings)
        
        print("✅ API client initialized successfully")
        print(f"📱 Phone Number ID: {settings.META_PHONE_NUMBER_ID}")
        print(f"🌐 API Base URL: {settings.GRAPH_API_URL}")
        
        # Test file path (you can change this to your test file)
        test_file_path = input("\n📁 Enter path to test media file (or press Enter to skip): ").strip()
        
        if not test_file_path:
            print("⏭️  Skipping file upload test")
            return
        
        # Remove quotes if present
        test_file_path = test_file_path.strip('"\'')
        
        # Check if file exists
        if not os.path.exists(test_file_path):
            print(f"❌ File not found: {test_file_path}")
            return
        
        print(f"\n📤 Uploading file: {os.path.basename(test_file_path)}")
        print("⏳ Please wait...")
        
        # Upload the media
        result = client.upload_media(test_file_path)
        
        if result:
            print("\n" + "=" * 50)
            print("✅ UPLOAD SUCCESSFUL!")
            print("=" * 50)
            print(f"📱 Media ID: {result['id']}")
            print(f"📂 File Name: {result['file_name']}")
            print(f"📊 Media Type: {result['media_type']}")
            print("\n💡 You can now use this Media ID in your WhatsApp templates!")
            
            # Show template usage example
            print("\n📝 Template Usage Example:")
            print("-" * 30)
            if result['media_type'] == 'video':
                print(f'''
{{
    "type": "header",
    "parameters": [
        {{
            "type": "video",
            "video": {{
                "id": "{result['id']}"
            }}
        }}
    ]
}}''')
            else:  # image
                print(f'''
{{
    "type": "header",
    "parameters": [
        {{
            "type": "image",
            "image": {{
                "id": "{result['id']}"
            }}
        }}
    ]
}}''')
            
            return result
        else:
            print("\n❌ Upload failed - no result returned")
            return None
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logger.error(f"Media upload test failed: {e}", exc_info=True)
        return None

def test_api_credentials():
    """Test if API credentials are properly configured"""
    print("\n🔐 Testing API Credentials")
    print("-" * 30)
    
    try:
        settings = get_settings()
        
        # Check required credentials
        required_vars = {
            'GRAPH_API_URL': settings.GRAPH_API_URL,
            'META_ACCESS_TOKEN': settings.META_ACCESS_TOKEN,
            'META_PHONE_NUMBER_ID': settings.META_PHONE_NUMBER_ID
        }
        
        all_present = True
        for var_name, var_value in required_vars.items():
            if var_value:
                print(f"✅ {var_name}: {'*' * min(len(str(var_value)), 20)}...")
            else:
                print(f"❌ {var_name}: Not set")
                all_present = False
        
        if all_present:
            print("\n✅ All required credentials are configured")
            return True
        else:
            print("\n❌ Some credentials are missing. Check your .env file.")
            return False
            
    except Exception as e:
        print(f"❌ Error checking credentials: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 WhatsApp Media Upload Test Suite")
    print("=" * 50)
    
    # Test credentials first
    if not test_api_credentials():
        print("\n⚠️  Cannot proceed without proper credentials")
        return
    
    # Test media upload
    result = test_media_upload()
    
    if result:
        print("\n🎉 All tests passed!")
        print("You can now use the media upload functionality in your application.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Test cancelled by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.error(f"Test script error: {e}", exc_info=True) 