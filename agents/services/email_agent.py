from langchain.agents import create_agent
from langchain_core.tools import tool
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv

load_dotenv()

# Email configuration from environment variables
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")

@tool
def send_simple_email(recipient: str, subject: str, body: str) -> str:
    """
    Send a simple text email.
    
    Args:
        recipient: Email address to send to
        subject: Email subject line
        body: Email message body
    
    Returns:
        Confirmation message
    """
    try:
        # Validate email format
        if "@" not in recipient or "." not in recipient:
            return f"Error: Invalid email address format: {recipient}"
        
        # Create message
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = SENDER_EMAIL
        msg['To'] = recipient
        
        # Send email via SMTP
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipient, msg.as_string())
        
        return f"✓ Email successfully sent to {recipient}"
    
    except smtplib.SMTPAuthenticationError:
        return "Error: SMTP authentication failed. Check email credentials."
    except smtplib.SMTPException as e:
        return f"Error: SMTP error occurred: {str(e)}"
    except Exception as e:
        return f"Error sending email: {str(e)}"

@tool
def send_html_email(recipient: str, subject: str, html_body: str, text_body: str = "") -> str:
    """
    Send an HTML formatted email with optional plain text fallback.
    
    Args:
        recipient: Email address to send to
        subject: Email subject line
        html_body: HTML formatted email body
        text_body: Plain text fallback (optional)
    
    Returns:
        Confirmation message
    """
    try:
        if "@" not in recipient or "." not in recipient:
            return f"Error: Invalid email address format: {recipient}"
        
        # Create multipart message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = SENDER_EMAIL
        msg['To'] = recipient
        
        # Attach text and HTML parts
        if text_body:
            msg.attach(MIMEText(text_body, 'plain'))
        msg.attach(MIMEText(html_body, 'html'))
        
        # Send email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipient, msg.as_string())
        
        return f"✓ HTML email successfully sent to {recipient}"
    
    except Exception as e:
        return f"Error sending HTML email: {str(e)}"

@tool
def send_bulk_email(recipients: str, subject: str, body: str) -> str:
    """
    Send an email to multiple recipients.
    
    Args:
        recipients: Comma-separated email addresses
        subject: Email subject line
        body: Email message body
    
    Returns:
        Confirmation with count of emails sent
    """
    try:
        # Parse recipients
        email_list = [email.strip() for email in recipients.split(",")]
        
        # Validate emails
        invalid_emails = [e for e in email_list if "@" not in e or "." not in e]
        if invalid_emails:
            return f"Error: Invalid email addresses: {', '.join(invalid_emails)}"
        
        # Send to each recipient
        sent_count = 0
        failed_count = 0
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            
            for recipient in email_list:
                try:
                    msg = MIMEText(body)
                    msg['Subject'] = subject
                    msg['From'] = SENDER_EMAIL
                    msg['To'] = recipient
                    
                    server.sendmail(SENDER_EMAIL, recipient, msg.as_string())
                    sent_count += 1
                except Exception as e:
                    failed_count += 1
                    print(f"Failed to send to {recipient}: {str(e)}")
        
        return f"✓ Bulk email complete: {sent_count} sent, {failed_count} failed"
    
    except Exception as e:
        return f"Error in bulk email: {str(e)}"

@tool
def send_email_with_cc_bcc(recipient: str, subject: str, body: str, cc: str = "", bcc: str = "") -> str:
    """
    Send an email with CC and BCC recipients.
    
    Args:
        recipient: Primary recipient email
        subject: Email subject line
        body: Email body
        cc: CC recipients (comma-separated)
        bcc: BCC recipients (comma-separated)
    
    Returns:
        Confirmation message
    """
    try:
        if "@" not in recipient or "." not in recipient:
            return f"Error: Invalid primary recipient: {recipient}"
        
        # Parse CC and BCC
        cc_list = [e.strip() for e in cc.split(",") if e.strip()] if cc else []
        bcc_list = [e.strip() for e in bcc.split(",") if e.strip()] if bcc else []
        
        # Create message
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = SENDER_EMAIL
        msg['To'] = recipient
        
        if cc_list:
            msg['Cc'] = ", ".join(cc_list)
        
        # Combine all recipients for sending
        all_recipients = [recipient] + cc_list + bcc_list
        
        # Send email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, all_recipients, msg.as_string())
        
        recipient_info = f"To: {recipient}"
        if cc_list:
            recipient_info += f", CC: {', '.join(cc_list)}"
        if bcc_list:
            recipient_info += f", BCC: {len(bcc_list)} recipient(s)"
        
        return f"✓ Email sent ({recipient_info})"
    
    except Exception as e:
        return f"Error sending email with CC/BCC: {str(e)}"

@tool
def verify_email_configuration() -> str:
    """
    Verify that email configuration is properly set up.
    
    Returns:
        Status message
    """
    checks = {
        "SMTP Server": SMTP_SERVER,
        "SMTP Port": SMTP_PORT,
        "Sender Email": SENDER_EMAIL if SENDER_EMAIL else "NOT SET",
        "Password": "SET" if SENDER_PASSWORD else "NOT SET"
    }
    
    if not all([SENDER_EMAIL, SENDER_PASSWORD]):
        return "⚠ Warning: Email credentials not fully configured. Set SENDER_EMAIL and SENDER_PASSWORD environment variables."
    
    status_msg = "✓ Email Configuration Status:\n"
    for key, value in checks.items():
        status_msg += f"  - {key}: {value}\n"
    
    return status_msg

def create_email_agent(model, checkpointer):
    """
    Create an email sending agent.
    
    Args:
        model: Language model to use (ChatOpenAI, ChatAnthropic, etc.)
        checkpointer: Optional checkpointer for persistence
    
    Returns:
        Compiled agent ready for email sending tasks
    """
    
    tools = [
        send_simple_email,
        send_html_email,
        send_bulk_email,
        send_email_with_cc_bcc,
        verify_email_configuration,
    ]
    
    system_prompt = """
You are an expert email management assistant. Your job is to help users send emails efficiently 
and professionally.

When a user asks you to send an email:
1. Clarify the recipient(s), subject, and message content
2. Determine if HTML formatting is needed
3. Check for CC/BCC requirements
4. Verify email addresses are valid
5. Use the appropriate tool to send the email
6. Confirm successful delivery

You have access to tools for:
- Sending simple text emails
- Sending HTML formatted emails
- Sending bulk emails to multiple recipients
- Sending emails with CC and BCC
- Verifying email configuration

Always ensure:
- Email addresses are valid before sending
- Subject lines are clear and descriptive
- Message content is professional and well-formatted
- Recipients are correctly specified
- Error messages are communicated clearly to the user

Before sending any email, verify that:
1. The recipient email address is valid
2. The subject line is appropriate
3. The message body is complete and makes sense
4. Any special requirements (CC, BCC, HTML) are requested
"""
    
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=checkpointer
    )
    
    return agent